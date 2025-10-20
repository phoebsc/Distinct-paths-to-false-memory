import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
import matplotlib
matplotlib.rcParams.update(new_rc_params)
import matplotlib.pyplot as plt
"""
0506 has USE for centrality and mpnet for prior knowledge
0529 has mpnet for both
"""
df = pd.read_excel('./C. script_LMM\data_sheets/semantic_measures_0618.xlsx')

def corrfunc(x, y, **kws):
    r, p = pearsonr(x, y)
    ax = plt.gca()

    # Determine significance stars
    if p < 0.001:
        stars = '***'
    elif p < 0.01:
        stars = '**'
    elif p < 0.05:
        stars = '*'
    else:
        stars = ''

    ax.annotate(f"r = {r:.2f}{stars}", xy=(0.95, 0.02), xycoords=ax.transAxes,
                ha='right', va='bottom', fontsize=11, color='black')

"""
plot dependent variables, collapsed across events
"""
main_color = '#97c5cf'  # Change this to your preferred color
line_color = '#271614'
dependent = ['conflict','confab','inference']
df_dependent = df[dependent+['story_id','event_id']].groupby(['story_id','event_id']).mean()
rename_dict = {
    'conflict': 'Factual error rate',
    'confab': 'Confabulation rate',
    'inference': 'Inference rate',
    # etc.
}

# Use the renamed version just for plotting
df_dependent = df_dependent.rename(columns=rename_dict)

cols = df_dependent.columns[:3]
pairwise = [(cols[0], cols[1]), (cols[1], cols[2]),(cols[2], cols[0])]

fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), constrained_layout=True)
title_fontsize = 10
label_fontsize = 8
tick_fontsize = 6.5

# Top row: histograms
# for i, col in enumerate(cols):
#     ax = axes[0,i]
#     sns.histplot(df_dependent[col], bins=15, color=main_color,ax=ax)
#     ax.set_title('')
#     ax.set_xlabel(col, fontsize=label_fontsize)
#     ax.set_ylabel("Count", fontsize=label_fontsize)
#     ax.tick_params(axis='both', labelsize=tick_fontsize)
#     ax.spines['left'].set_linewidth(0.6)
#     ax.spines['bottom'].set_linewidth(0.6)
#     ax.spines[['right', 'top']].set_visible(False)

#correlation plots
for j, (xcol, ycol) in enumerate(pairwise):
    ax = axes[j]
    sns.regplot(
        data=df_dependent, x=xcol, y=ycol, ax=ax,
        line_kws={'color': line_color,'lw':1.5}, scatter_kws={'color': main_color, 's': 10}
    )
    ax.set_xlabel(xcol, fontsize=label_fontsize)
    ax.set_ylabel(ycol, fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines[['right', 'top']].set_visible(False)

    r_val, p_val = pearsonr(df_dependent[xcol], df_dependent[ycol])
    p_val = 3 * p_val  # Bonferroni correction
    if p_val < 0.001:
        stars = '***'
    elif p_val < 0.01:
        stars = '**'
    elif p_val < 0.05:
        stars = '*'
    else:
        stars = ''

    ax.text(
        0.95, 0.05, fr"$\it{{r}}$ = {r_val:.2f}{stars}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=label_fontsize,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
    )

plt.tight_layout()
plt.show()
plt.savefig("./figures\svg/supp/corr_outcome.svg")

"""
plot independent variables, collapsed across events
"""
main_color2 = '#F1CBA9'
independ = ['centrality','PPL_context', 'prior']
df_ind = df[independ+['story_id','event_id']].groupby(['story_id','event_id']).mean()

rename_dict = {
    'centrality': 'Semantic centrality',
    'PPL_context': 'Surprisal',
    'prior': 'Similarity to narrative corpus',
    # etc.
}

# Use the renamed version just for plotting
df_ind = df_ind.rename(columns=rename_dict)

cols = df_ind.columns[:3]
pairwise = [(cols[0], cols[1]), (cols[1], cols[2]),(cols[2], cols[0])]

fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), constrained_layout=True)
title_fontsize = 10
label_fontsize = 8
tick_fontsize = 6.5

# Top row: histograms
# for i, col in enumerate(cols):
#     ax = axes[0,i]
#     sns.histplot(df_ind[col], bins=15, color=main_color2,ax=ax)
#     ax.set_title('')
#     ax.set_xlabel(col, fontsize=label_fontsize)
#     ax.set_ylabel("Count", fontsize=label_fontsize)
#     ax.tick_params(axis='both', labelsize=tick_fontsize)
#     ax.spines['left'].set_linewidth(0.6)
#     ax.spines['bottom'].set_linewidth(0.6)
#     ax.spines[['right', 'top']].set_visible(False)

# Bottom row: correlation plots with lmplot
for j, (xcol, ycol) in enumerate(pairwise):
    # Create lmplot separately
    ax_lm = axes[j]
    sns.regplot(
        data=df_ind, x=xcol, y=ycol, ax=ax_lm,
        line_kws={'color': line_color,'lw':1.5}, scatter_kws={'color': main_color2, 's': 10}
    )
    # Set ranges, title, labels
    ax_lm.set_title('')
    ax_lm.set_xlabel(xcol, fontsize=label_fontsize)
    ax_lm.set_ylabel(ycol, fontsize=label_fontsize)
    ax_lm.tick_params(axis='both', labelsize=tick_fontsize)
    ax_lm.spines['left'].set_linewidth(0.6)
    ax_lm.spines['bottom'].set_linewidth(0.6)
    ax_lm.spines[['right', 'top']].set_visible(False)
    r_val, p_val = pearsonr(df_ind[xcol], df_ind[ycol])
    # correction
    p_val = 3 * p_val
    # Determine significance stars
    if p_val < 0.001:
        stars = '***'
    elif p_val < 0.01:
        stars = '**'
    elif p_val < 0.05:
        stars = '*'
    else:
        stars = ''

    # Add correlation coefficient (italicized 'r', 2 decimals)
    ax_lm.text(0.95, 0.05, fr"$\it{{r}}$ = {r_val:.2f}{stars}",
               transform=ax_lm.transAxes, ha="right", va="bottom",
               fontsize=label_fontsize,
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
plt.tight_layout()
plt.savefig("./figures\svg/supp/corr_ind.svg")

