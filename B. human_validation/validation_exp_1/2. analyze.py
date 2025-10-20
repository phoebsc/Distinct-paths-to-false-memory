import json
import numpy as np
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
path = './D. human_validation/validation_exp_Sept2024\data/false-memory-rating-default-rtdb-export-1014.json'
with open(path,  encoding='utf-8') as f:
    d = json.load(f)
path2 = './D. human_validation/validation_exp_March2025\data/false-memory-rating-default-rtdb-export-0610.json'
with open(path2,  encoding='utf-8') as f:
    d2 = json.load(f)
# combining data from Sept 2024
# 041 was removed due 2 in understanding
subject_list1 = ['002','003','004','005','006','007','008','009','010']
subject_list2 = ['0'+str(i) for i in range(22,41)]+['042']
subject_list_first = ['002','003','004','005','006','007','008','009','010','039','040','042','045','046']
subject_list_last = ['0'+str(i) for i in range(22,39)]

data1 = [sub for sub in d.items() if sub[0] in subject_list1]
data2 = [sub for sub in d2.items() if sub[0] in subject_list2]
data = data1 + data2

data_first = [sub for sub in data if sub[0] in subject_list_first]
data_last = [sub for sub in data if sub[0] in subject_list_last]
story_ids = ['pieman','eyespy','oregontrail','baseball']
"""
load
this block should generate:
    pieman_answers: an array of (9, 30, 2). 
        Slice with (index_subject, index_trial, 0) to get yes/no answers (yes=0, no=1),
        Slice with (index_subject, index_trial, 1) to get confidence ratings (0-4)
    eyespy_answers: an array of (9, 30, 2). same as above
    all_answers: an array of (9, 60, 2), where the above two are concatenated on the second axis.
    
agree = 0; disagree =1
"""
pieman_answers = []
eyespy_answers = []
for sub in data_first:
    trials = [t for t in sub[1]['data'] if 'id' in list(t.keys())]
    print(sub[0], sub[1]['data'][104]['response'])  # check questionnaire
    # pieman
    pieman = trials[0:30]
    answers = np.zeros((30,2))
    for trial in pieman:
        answers[trial['id']] = trial['response']['truth'], trial['response']['confidence']
    pieman_answers.append(answers)
    # eyespy
    eyespy = trials[30:]
    answers = np.zeros((30,2))
    for trial in eyespy:
        answers[trial['id']] = trial['response']['truth'], trial['response']['confidence']
    eyespy_answers.append(answers)
pieman_answers = np.array(pieman_answers)
eyespy_answers = np.array(eyespy_answers)
"""
load last two
"""
oregon_answers = []
baseball_answers = []
for sub in data_last:
    trials = [t for t in sub[1]['data'] if 'id' in list(t.keys())]
    print(sub[0], sub[1]['data'][104]['response'])  # check questionnaire
    # pieman
    oregon = trials[0:30]
    answers = np.zeros((30,2))
    for trial in oregon:
        answers[trial['id']] = trial['response']['truth'], trial['response']['confidence']
    oregon_answers.append(answers)
    # eyespy
    baseball = trials[30:]
    answers = np.zeros((30,2))
    for trial in baseball:
        answers[trial['id']] = trial['response']['truth'], trial['response']['confidence']
    baseball_answers.append(answers)
oregon_answers = np.array(oregon_answers)
baseball_answers = np.array(baseball_answers)
all_answers = np.concatenate([pieman_answers,eyespy_answers], axis=1)
"""
demographic
"""
year = []
gender = []
for sub in data_first+data_last:
    try:
        gender.append(sub[1]['data'][1]['response']['gender'])
        year.append(2024-int(sub[1]['data'][1]['response']['birth_year']))
    except:
        print(sub[0])
"""
compute mean and SE agreement
"""
from itertools import combinations
def make_hh_agreement_matrix(human_labels):
    """
    human_labels: N x R array of 0/1 or True/False indicating
                  each human rater's label on each item.

    Returns:
      hh_data: N x P array of True/False,
               where P = number of rater pairs = R*(R-1)/2.
               Each column is whether that pair of raters agreed
               for each item.
    """
    N, R = human_labels.shape

    # List to hold each pair's T/F column
    hh_cols = []

    # Generate all unique rater pairs
    for (r1, r2) in combinations(range(R), 2):
        # Compare rater r1 vs r2 across all N items
        agreement_bool = (human_labels[:, r1] == human_labels[:, r2])
        hh_cols.append(agreement_bool)

    # Stack all pair-agreement columns side by side
    # hh_cols is a list of N-element boolean arrays
    # shape becomes (N, P) after column_stack
    hh_data = np.column_stack(hh_cols)

    return hh_data
def bootstrap_compare(ai_data, hh_data, B=1000, random_state=None):
    """
    ai_data: NxR boolean (True/False) for AI–human
    hh_data: NxP boolean (True/False) for human–human pairs
    B: number of bootstrap iterations

    Returns a dict with observed values, plus bootstrap distributions.
    """
    if random_state is not None:
        np.random.seed(random_state)

    N = ai_data.shape[0]

    # Observed means
    obs_ai_human = np.mean(ai_data)
    obs_hh = np.mean(hh_data)
    obs_diff = obs_ai_human - obs_hh

    dist_ai_human = np.zeros(B)
    dist_hh = np.zeros(B)

    for b in range(B):
        # Resample row indices with replacement
        boot_idx = np.random.randint(0, N, size=N)

        ai_boot = ai_data[boot_idx, :]
        hh_boot = hh_data[boot_idx, :]

        dist_ai_human[b] = np.mean(ai_boot)
        dist_hh[b] = np.mean(hh_boot)

    dist_diff = dist_ai_human - dist_hh

    return {
        'obs_ai_human': obs_ai_human,
        'obs_hh': obs_hh,
        'obs_diff': obs_diff,
        'dist_ai_human': dist_ai_human,
        'dist_hh': dist_hh,
        'dist_diff': dist_diff
    }
def get_bootstrap_ci(distribution, alpha=0.05):
    lower = np.percentile(distribution, 100 * (alpha / 2))
    upper = np.percentile(distribution, 100 * (1 - alpha / 2))
    return (lower, upper)

def two_sided_bootstrap_p_value(obs_diff, dist_diff):
    """
    obs_diff: the observed difference (AI–Human minus Human–Human)
    dist_diff: 1D array of bootstrap differences
    Returns a 2-sided p-value based on how often the bootstrap differences cross 0.
    """
    if obs_diff > 0:
        # fraction of bootstrap differences that are <= 0
        p_one_sided = np.mean(dist_diff <= 0)
    else:
        # fraction of bootstrap differences that are >= 0
        p_one_sided = np.mean(dist_diff >= 0)

    p_two_sided = 2 * p_one_sided
    if p_two_sided > 1:
        p_two_sided = 1.0

    return p_two_sided

# Split between acc/inacc trials
pieman_ground = np.array(pd.read_excel('./D. human_validation/validation_exp_Sept2024/trial_excels/pieman.xlsx')['false_mem'])
eyespy_ground = np.array(pd.read_excel('./D. human_validation/validation_exp_Sept2024/trial_excels/eyespy.xlsx')['false_mem'])
oregon_ground = np.array(pd.read_excel('./D. human_validation/validation_exp_Sept2024/trial_excels/oregontrail.xlsx')['false_mem'])
baseball_ground = np.array(pd.read_excel('./D. human_validation/validation_exp_Sept2024/trial_excels/baseball.xlsx')['false_mem'])

# individual stories
means = np.zeros((4,2,2))  # stories, accurate/inaccurate, ai/human
SEs = np.zeros((4,2,2))
agg_dists = {
    'acc': {'ai': [], 'hh': [], 'diff': [], 'obs_diff': []},   # <-- store obs_diff
    'inacc':   {'ai': [], 'hh': [], 'diff': [], 'obs_diff': []}
}
for i, (ground, answers) in enumerate(zip([pieman_ground,eyespy_ground,oregon_ground,baseball_ground],
                                          [pieman_answers,eyespy_answers,oregon_answers,baseball_answers])):
    # accurate
    ratings_acc = 1 - answers[:,~ground,0]  # person x questions
    ratings_inacc = 1 - answers[:,ground,0]  # person x questions

    for r, ratings in enumerate([ratings_acc,ratings_inacc]):
        condition = ['acc','inacc'][r]
        ai_data = ratings
        hh_data = make_hh_agreement_matrix(ratings)
        res = bootstrap_compare(ai_data, hh_data, B=2000, random_state=42)
        ci_ai = get_bootstrap_ci(res['dist_ai_human'])
        ci_hh = get_bootstrap_ci(res['dist_hh'])
        ci_diff = get_bootstrap_ci(res['dist_diff'])
        p_val = two_sided_bootstrap_p_value(res['obs_diff'], res['dist_diff'])
        # NEW: store the distributions + observed per-story difference
        agg_dists[condition]['ai'].append(res['dist_ai_human'])
        agg_dists[condition]['hh'].append(res['dist_hh'])
        agg_dists[condition]['diff'].append(res['dist_diff'])
        agg_dists[condition]['obs_diff'].append(res['obs_diff'])
        # add data
        SEs[i, r,:] = (ci_ai[1] - ci_ai[0]) / 2 / 1.96, (ci_hh[1] - ci_hh[0]) / 2 / 1.96
        means[i, r, :] = res['obs_ai_human'], res['obs_hh']
        print(story_ids[i], ['accurate','inacc'][r])
        print("AI–Human agreement: obs =", np.round(res['obs_ai_human'],3), ", 95% CI =", np.round(ci_ai,3))
        print("Human–Human agreement: obs =", np.round(res['obs_hh'],3), ", 95% CI =", np.round(ci_hh,3))
        print("Difference: obs =", np.round(res['obs_diff'],3), ", 95% CI = ", np.round(ci_diff,3))
        print("P-value for difference =", np.round(p_val,3))

# === NEW: aggregated Option-B CIs per condition (averaging bootstrap draws across stories) ===
for cond_i, condition in enumerate(['acc','inacc']):
    # stack to (n_stories, B)
    dist_ai_mat  = np.stack(agg_dists[condition]['ai'],   axis=0)
    dist_hh_mat  = np.stack(agg_dists[condition]['hh'],   axis=0)
    dist_dif_mat = np.stack(agg_dists[condition]['diff'], axis=0)

    # average across stories per bootstrap iteration -> (B,)
    comb_ai   = dist_ai_mat.mean(axis=0)
    comb_hh   = dist_hh_mat.mean(axis=0)
    comb_diff = dist_dif_mat.mean(axis=0)  # equivalent to comb_ai - comb_hh

    # CIs from combined distributions
    ci_ai_comb   = get_bootstrap_ci(comb_ai)
    ci_hh_comb   = get_bootstrap_ci(comb_hh)
    ci_diff_comb = get_bootstrap_ci(comb_diff)

    # observed combined means (to report alongside)
    obs_ai_comb = np.mean(means[:,cond_i,0])
    obs_hh_comb = np.mean(means[:,cond_i,1])
    obs_diff_comb = obs_ai_comb - obs_hh_comb

    print(f"\n=== Aggregated across stories: {condition} ===")
    print("AI–Human aggregate:   obs =", np.round(obs_ai_comb,3),
          ", 95% CI =", np.round(ci_ai_comb,3))
    print("Human–Human aggregate: obs =", np.round(obs_hh_comb,3),
          ", 95% CI =", np.round(ci_hh_comb,3))
    print("Difference aggregate:  obs =", np.round(obs_diff_comb,3),
          ", 95% CI =", np.round(ci_diff_comb,3))

    # === NEW: Hierarchical bootstrap p-value for the mean *difference* (your recipe) ===
    # Use the per-story difference distributions we already collected.
    dist_diffs   = agg_dists[condition]['diff']
    obs_diffs    = agg_dists[condition]['obs_diff']

    # Infer B from stored dists to avoid extra globals
    B_local = dist_diffs[0].shape[0]
    np.random.seed(42)
    boot_means = np.empty(B_local)
    for b in range(B_local):
        # pick one bootstrap draw per story and average
        draws = [dist[np.random.randint(0, B_local)] for dist in dist_diffs]
        boot_means[b] = np.mean(draws)

    obs_mean = float(np.mean(obs_diffs))
    ci_mean  = get_bootstrap_ci(boot_means)

    # two‐sided p‐value for the mean effect
    if obs_mean > 0:
        p_one = np.mean(boot_means <= 0)
    else:
        p_one = np.mean(boot_means >= 0)
    p_two = min(1.0, 2 * p_one)

    print(f"\n=== [{condition.upper()}] Hierarchical Bootstrap (Difference) ===")
    print(f"Mean observed effect: {obs_mean:.3f}")
    print(f"95% CI on mean effect: ({ci_mean[0]:.3f}, {ci_mean[1]:.3f})")
    print(f"Two‐sided p-value: {p_two:.3f}")
"""
stats across stories
"""
conds = ['acc', 'inacc']
B = 2000
for condition in conds:
    # storage for this condition
    obs_diffs = []
    dist_diffs = []
    for i, (ground, answers) in enumerate(zip([pieman_ground,eyespy_ground,oregon_ground,baseball_ground],
                                              [pieman_answers,eyespy_answers,oregon_answers,baseball_answers])):
        if condition =='acc':
            ratings = 1 - answers[:,~ground,0]  # person x questions
        else:
            ratings = 1 - answers[:,ground,0]  # person x questions
        ai_data = ratings
        hh_data = make_hh_agreement_matrix(ratings)
        res = bootstrap_compare(ai_data, hh_data, B=2000, random_state=42)
        ci_ai = get_bootstrap_ci(res['dist_ai_human'])
        ci_hh = get_bootstrap_ci(res['dist_hh'])
        ci_diff = get_bootstrap_ci(res['dist_diff'])
        p_val = two_sided_bootstrap_p_value(res['obs_diff'], res['dist_diff'])
        # append
        obs_diffs.append(res['obs_diff'])
        dist_diffs.append(res['dist_diff'])
        # 2) Hierarchical bootstrap across stories for this condition
    np.random.seed(42)
    boot_means = np.empty(B)
    for b in range(B):
        # pick one bootstrap draw per story and average
        draws = [dist[np.random.randint(0, B)] for dist in dist_diffs]
        boot_means[b] = np.mean(draws)

    obs_mean = np.mean(obs_diffs)
    ci_mean = get_bootstrap_ci(boot_means)

    # two‐sided p‐value for the mean effect
    if obs_mean > 0:
        p_one = np.mean(boot_means <= 0)
    else:
        p_one = np.mean(boot_means >= 0)
    p_two = min(1.0, 2 * p_one)

    print(f"\n=== [{condition.upper()}] Hierarchical Bootstrap ===")
    print(f"Mean observed effect: {obs_mean:.3f}")
    print(f"95% CI on mean effect: ({ci_mean[0]:.3f}, {ci_mean[1]:.3f})")
    print(f"Two‐sided p-value: {p_two:.3f}\n")

"""
compare the groups (between true and false trials)
"""

B = 2000

# 1) Collect per-story results
obs_diff = {c: [] for c in conds}
dist_diff = {c: [] for c in conds}

for condition in conds:
    # storage for this condition
    obs_diffs = []
    dist_diffs = []
    for i, (ground, answers) in enumerate(zip([pieman_ground, eyespy_ground, oregon_ground, baseball_ground],
                                              [pieman_answers, eyespy_answers, oregon_answers, baseball_answers])):
        if condition == 'acc':
            ratings = 1 - answers[:, ~ground, 0]  # person x questions
        else:
            ratings = 1 - answers[:, ground, 0]  # person x questions
        R = ratings
        res = bootstrap_compare(R, make_hh_agreement_matrix(R), B=B, random_state=42)

        obs_diff[condition].append(res['obs_diff'])
        dist_diff[condition].append(res['dist_diff'])

# 2) Hierarchical bootstrap for “difference of gaps”
np.random.seed(42)
boot_deltas = np.empty(B)
for b in range(B):
    # sample one draw per story for each condition
    mean_conflict = np.mean([dist[np.random.randint(0,B)] for dist in dist_diff['acc']])
    mean_confab = np.mean([dist[np.random.randint(0,B)] for dist in dist_diff['inacc']])
    boot_deltas[b] = mean_conflict - mean_confab

# 3) Observed “difference of means”
obs_delta = np.mean(obs_diff['acc']) - np.mean(obs_diff['inacc'])

# 4) p-value and CI
if obs_delta > 0:
    p_one = np.mean(boot_deltas <= 0)
else:
    p_one = np.mean(boot_deltas >= 0)
p_two = min(1, 2 * p_one)
ci_delta = get_bootstrap_ci(boot_deltas)

print("=== Condition‐Gap Comparison ===")
print(f"Observed Δ = {obs_delta:.3f}")
print(f"95% CI = ({ci_delta[0]:.3f}, {ci_delta[1]:.3f})")
print(f"Two‐sided p = {p_two:.3f}")
