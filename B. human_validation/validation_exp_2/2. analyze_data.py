import json
import pandas as pd
import numpy as np
filename = './D. human_validation/validation_exp_Feb2025/data/storycomprehension-default-rtdb-export_0327.json'
roster = pd.read_csv('./D. human_validation/validation_exp_Feb2025/data/roster_download_1644.csv',header=0)
roster['survey_id'] = roster['survey_id'].astype(str)
roster_names = roster['survey_id'].to_list()
f = open(filename,)
data = json.load(f)
mapping_trial_number = {'pieman':7,'eyespy':9,'oregontrail':9,'baseball':9}
mapping_length = {'pieman':8*60+22,'eyespy':13*60 ,'oregontrail':12*60+23,'baseball':12*60+48}

# read data
subs = [x for x in list(data.keys()) if x in roster_names]
df = pd.DataFrame(columns=['participant','story','question_id','question','answer'])
n=0
sub_questions = []
time_elapsed = []
for sub in subs:
    sub_data = data[sub]['data']
    # append questionnaire data
    assert list(sub_data[-2]['response'].keys()) == ['difficult', 'engage', 'understood']
    sub_questions.append(list(sub_data[-2]['response'].values()))
    # audio data
    audios = [t['stimulus'].split('/')[-1].split('_')[0] for t in sub_data if 'audio' in t['trial_type']][1:]
    # trial data
    trials = [t for t in sub_data if 'question' in t.keys()]
    assert len(trials) == mapping_trial_number[audios[0]]+mapping_trial_number[audios[1]]
    for i, trial in enumerate(trials):
        for q, a in zip(list(trial['response'].keys()), list(trial['response'].values())):
            prompt = [question['prompt'] for question in trial['question'] if question['name']==q][0]
            # determine which story
            if i < mapping_trial_number[audios[0]]:
                story = audios[0]
            else:
                story = audios[1]
            df.loc[n] = [sub, story, q, prompt, a]
            n+=1
    # append experiment length in time
    time_elapsed.append(sub_data[-2]['time_elapsed'] - sub_data[0]['time_elapsed']\
                      -mapping_trial_number[audios[0]]*1000-mapping_trial_number[audios[1]]*1000)
# add the ground truth (AI answers)
df_original = pd.DataFrame(columns=['story','question_id','ground','flag'])
story_ids = ['pieman','eyespy','oregontrail','baseball']
for story_id in story_ids:
    trial_excel = pd.read_excel('./D. human_validation/validation_exp_Feb2025/trial_excels/%s_agreement.xlsx' % story_id)
    trial_excel['question_id'] = 'excel_id'+trial_excel['ID'].astype(int).astype(str)
    trial_excel['story'] = story_id
    trial_excel['ground'] = trial_excel['Ground']
    trial_excel['flag'] = trial_excel['Flag']
    df_original = pd.concat([df_original,trial_excel[['story','question_id','ground','flag','type of error']]])
df = pd.merge(df,df_original,how='left',on=['story', 'question_id'])
df['ground'] = df['ground'].astype(str)
df['agreeAI'] = df['answer'].str.strip()==df['ground'].str.strip()
# remove trials that had an error and did not record answers
df = df[df['answer'].str.len()>1]
"""
check questionnaire (difficulty, engagement, understood)
"""

cutoff = 2  # 2 = "neutral", 3 = "agree"
sub_questions = np.array(sub_questions)
# removal
rm1 = np.where(sub_questions[:, 1] <= cutoff)[0]
rm2 = np.where(sub_questions[:, 2] <= cutoff)[0]

"""
check attention trials
"""
df['qid_num'] = df['question_id'].str.extract(r'(\d+)$').astype(int)
# Filter rows with question_id >= 50
filtered = df[df['qid_num'] >= 50]
# Group by participant and compute mean of agree
result = filtered.groupby('participant')['agreeAI'].mean().reset_index()
rm3 = result.loc[result.agreeAI<=0.9,'participant'].to_list()

remove = np.concatenate([np.array(subs)[rm1],
                         np.array(subs)[rm2],
                         rm3])
print('total N', len(subs), 'removed N', len(set(remove)))
df = df.loc[~df.participant.isin(remove)]
df.groupby('story')['participant'].nunique()
# remove attention trials
df = df[df['qid_num'] < 50]

# remove some trials due to errors
df = df.loc[~((df.story=='oregontrail') & (df.question_id.astype(str).str.endswith(('35', '41'))))]

"""
demographic after exclusion
"""
par_list = pd.read_csv('./D. human_validation/validation_exp_Feb2025/data/prescreen_data.csv')
roster = pd.read_csv('./D. human_validation/validation_exp_Feb2025/data/roster_download_1644.csv')[['login_id','survey_id']]
par_list = pd.merge(par_list,roster,how='left',on='login_id')
par_list = par_list.loc[~par_list.survey_id.isna()]
par_list.survey_id = par_list.survey_id.astype(int).astype(str)
sub_list = par_list.loc[par_list.survey_id.isin(np.unique(df.participant))]

sub_list['age'] = 2025 - pd.to_numeric(sub_list['BirthDate'].str.split('/', expand=True)[2], errors='coerce')
age = [x for x in sub_list['age'].to_list() if x>0]
print('age mean', np.mean(age), 'std', np.std(age))
print('gender', sub_list['GenderNow'].value_counts())
"""
check agreements with AI
"""
def agreement(subdf):
    subdf['flag'] = subdf['flag'].replace({'x': 1}).fillna(0)

    return pd.Series(dict(agree=np.sum(subdf['agreeAI'])/len(subdf),
                          subn=len(subdf),
                          flag=np.mean(subdf['flag']),
                          type_of_error=subdf['type of error'].to_list()[0]))

# simple agreement
df_agree = df.groupby(['story','question_id']).apply(agreement)
df_agree['agree'].mean()  # avg all ppl

# split errors
for story_id in story_ids:
    print(story_id,df_agree.loc[story_id].groupby('type_of_error')['agree'].mean())
df_agree_per_story = df.groupby(['story','type of error']).apply(agreement)
print('averaging four stories')
print('confab',(df_agree_per_story.iloc[0,0]+df_agree_per_story.iloc[2,0]+df_agree_per_story.iloc[4,0]+
                     df_agree_per_story.iloc[6,0])/4)
print('conflict',(df_agree_per_story.iloc[1,0]+df_agree_per_story.iloc[3,0]+df_agree_per_story.iloc[5,0]+
                       df_agree_per_story.iloc[7,0])/4)
print("averaging all ppl", df_agree.groupby('type_of_error')['agree'].mean())

# group errors
story_agreement = []
for story_id in story_ids:
    agg = df_agree.loc[story_id]['agree'].mean()
    print(story_id,agg)
    story_agreement.append(agg)
# avg 4 stories
print('averaging four stories', np.mean(story_agreement), "averaging all ppl", df_agree['agree'].mean())

# for story_id in story_ids:
#     plt.figure()
#     plt.bar(np.arange(len(df_agree.loc[story_id]['agree'])),df_agree.loc[story_id]['agree'])
#     plt.scatter(np.arange(len(df_agree.loc[story_id]['agree'])),df_agree.loc[story_id]['flag'])
#     plt.title(story_id)
#     plt.ylabel('agreement with AI')
#     plt.xlabel('questions')
#     plt.show()
#     print(df_agree.loc[story_id]['agree'].mean())

"""
correlation between subjects on their answers
"""
def sub_simpleMatch(subdf):
    df_wide = subdf.pivot(index='question_id', columns='participant', values='agreeAI')
    ratings = df_wide.to_numpy().T.astype(int)

    n_raters = ratings.shape[0]
    pairwise_smc_values = {}
    # Compute pairwise SMC for each (i, j) pair of raters
    for i, j in combinations(range(n_raters), 2):
        rater_i = ratings[i]
        rater_j = ratings[j]
        # SMC is the fraction of items where rater_i == rater_j
        smc_value = np.mean(rater_i == rater_j)
        pairwise_smc_values[(i, j)] = smc_value
    # Compute the average of all pairwise SMC values
    avg_smc = np.mean(list(pairwise_smc_values.values()))
    return avg_smc
df_agree_sub = df.groupby(['story']).apply(sub_simpleMatch)
df_agree_sub.mean(axis=0)

# split into errors
df_agree_sub = df.groupby(['story','type of error']).apply(sub_simpleMatch)
print('confab_mean',(df_agree_sub.iloc[0]+df_agree_sub.iloc[2]+df_agree_sub.iloc[4]+df_agree_sub.iloc[6])/4)
print('conflict_mean',(df_agree_sub.iloc[1]+df_agree_sub.iloc[3]+df_agree_sub.iloc[5]+df_agree_sub.iloc[7])/4)

print('averaging four stories', df_agree_sub.mean(axis=0))


"count number of unique trials"
confab = []
conflict = []
for story_id in story_ids:
    trial_excel = pd.read_excel('./D. human_validation/validation_exp_Feb2025/trial_excels/%s_agreement.xlsx' % story_id)
    conflict.append(trial_excel['type of error'].value_counts()['conflict'])
    confab.append(trial_excel['type of error'].value_counts()['confab'])

print('confab trials',np.sum(confab),'conflict trials',np.sum(conflict))

"""
boostrapping for significance and CI in plots
note: CI overlap is not a reliable way to check significance,
so we have two CIs for plotting, and a CI for the difference
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

# individual stories
for story in story_ids:
    subdf = df.loc[df.story==story]
    df_wide = subdf.pivot(index='question_id', columns='participant', values='agreeAI')
    ratings = df_wide.to_numpy()
    ai_data = ratings
    hh_data = make_hh_agreement_matrix(ratings)
    res = bootstrap_compare(ai_data, hh_data, B=2000, random_state=42)

    ci_ai = get_bootstrap_ci(res['dist_ai_human'])
    ci_hh = get_bootstrap_ci(res['dist_hh'])
    ci_diff = get_bootstrap_ci(res['dist_diff'])
    p_val = two_sided_bootstrap_p_value(res['obs_diff'], res['dist_diff'])
    print(story)
    print("AI–Human agreement: obs =", np.round(res['obs_ai_human'],3), ", 95% CI =", np.round(ci_ai,3))
    print("Human–Human agreement: obs =", np.round(res['obs_hh'],3), ", 95% CI =", np.round(ci_hh,3))
    print("Difference: obs =", np.round(res['obs_diff'],3), ", 95% CI = ", np.round(ci_diff,3))
    print("P-value for difference =", np.round(p_val,3))

"""
compute confab vs conflict, grouping all stories
"""
errors = np.zeros((4,2,2))  # stories, [conflict, confab], [ai_human, human-human]
means = np.zeros((4,2,2))
pvals = np.zeros((4,2))  # stories, [conflict, confab]
agg_dists = {
    'conflict': {'ai': [], 'hh': [], 'diff': [], 'obs_diff': []},   # <-- store obs_diff
    'confab':   {'ai': [], 'hh': [], 'diff': [], 'obs_diff': []}
}
for s,story in enumerate(story_ids):
    for c,condition in enumerate(['conflict', 'confab']):
        subdf = df.loc[(df.story==story) & (df['type of error'] == condition)]
        df_wide = subdf.pivot(index='question_id', columns='participant', values='agreeAI')
        ratings = df_wide.to_numpy()
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
        means[s,c] = [np.round(res['obs_ai_human'],3),np.round(res['obs_hh'],3)]
        errors[s,c] = (ci_ai[1]-ci_ai[0])/2/1.96, (ci_hh[1]-ci_hh[0])/2/1.96
        pvals[s,c] = p_val
        print(story, condition)
        print("AI–Human agreement: obs =", np.round(res['obs_ai_human'],3), ", 95% CI =", np.round(ci_ai,3))
        print("Human–Human agreement: obs =", np.round(res['obs_hh'],3), ", 95% CI =", np.round(ci_hh,3))
        print("Difference: obs =", np.round(res['obs_diff'],3), ", 95% CI = ", np.round(ci_diff,3))
        print("P-value for difference =", np.round(p_val,3))

"""
aggregated CIs and comparisons
"""
# === NEW: aggregated Option-B CIs per condition (averaging bootstrap draws across stories) ===
for cond_i, condition in enumerate(['conflict','confab']):
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
