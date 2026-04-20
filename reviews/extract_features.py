import os, re
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

project_root = "/Users/fadechen/Documents/GitHub/Distinct-paths-to-false-memory"  # works if notebook is at root

def find_index_for_sum(nums, target_sum):
    current_sum = 0
    for index, num in enumerate(nums):
        current_sum += num
        if current_sum == target_sum:
            return index
    return -1  # Return -1 if the target sum is not reached

story_ids = ['pieman','eyespy','oregontrail','baseball']
model_ids = ['mpnet', 'mpnet','gpt'] # prior knowledge, centrality, surprisal

df_prior = pd.DataFrame(columns=['story_id', 'event_id', 'event_text','prior'])
keep_ids = np.load(project_root+'/semantic_features/keep_ids.npy',allow_pickle=True)
with open(project_root+'/semantic_features/PriorKnowledge_allstories_%s.pkl' % model_ids[0], 'rb') as file:
    pickled_prior=pickle.load(file)
n=0
for j, story_id in enumerate(story_ids):
    story_segs = pd.read_excel(project_root+'/A. script_prompting/story_segs.xlsx',
                               sheet_name='%s_segs' % story_id)['events'].to_list()
    story_segs = np.array(story_segs)[keep_ids[j]]
    # features of the sentences
    prior = pickled_prior[j]
    # add data
    for k in range(len(prior)):  # event
        event_id = find_index_for_sum(keep_ids[j], k+1)
        df_prior.loc[n] = [story_id, event_id, story_segs[k], prior[k]]
        n+=1

df_prior.to_excel(project_root+'/reviews/data/extract_features_prior.xlsx')
