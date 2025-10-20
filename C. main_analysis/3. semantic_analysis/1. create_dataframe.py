import os, re
import pandas as pd
import numpy as np
import pickle
base_path = 'C:/Users\hchen\Dropbox\PycharmProjects/false_mem'
def check_correct(t, c):
    return np.nan if np.isnan(t) else (t == 1 and c == 0)
# Vectorize the function to apply it element-wise
vectorize_correct = np.vectorize(check_correct,otypes=[object])

def check_false(t, c):
    return np.nan if np.isnan(t) else (t == 0 or c == 1)
# Vectorize the function to apply it element-wise
vectorize_false = np.vectorize(check_false,otypes=[object])

def find_index_for_sum(nums, target_sum):
    current_sum = 0
    for index, num in enumerate(nums):
        current_sum += num
        if current_sum == target_sum:
            return index
    return -1  # Return -1 if the target sum is not reached

story_ids = ['pieman','eyespy','oregontrail','baseball']
model_ids = ['mpnet', 'mpnet','gpt'] # prior knowledge, centrality, surprisal
axis = 0  # 0=event, 1=subject
arr_perc_false = []
arr_perc_correct = []
arr_perc_conflict = []
arr_perc_overlap = []
arr_perc_confab = []
arr_perc_inference = []
df_all = pd.DataFrame(columns=['story_id', 'par_id', 'event_id', 'centrality', 'PPL_context', 'prior',
                                 'recall', 'conflict', 'confab', 'correct','false','inference','length'])
keep_ids = np.load(base_path+'/E. semantic_features/keep_ids.npy',allow_pickle=True)
with open(base_path+'/E. semantic_features/PriorKnowledge_allstories_%s.pkl' % model_ids[0], 'rb') as file:
    pickled_prior=pickle.load(file)
n=0
for j, (story_id, subfolder) in enumerate(zip(story_ids)):
    recall_path = './GPT_output/memory_classification/%s' % story_id  # the already rated recall transcripts
    recall_paths = [recall_path + '/'+x
                    for x in os.listdir(recall_path)]
    story_segs = pd.read_excel('./A. script_prompting/story_segs.xlsx',
                               sheet_name='%s_segs' % story_id)['events'].to_list()
    story_segs = np.array(story_segs)[keep_ids[j]]
    # features of the sentences
    centrality = np.load(base_path+'/E. semantic_features/centrality_%s_%s.npy' % (model_ids[1],story_id))
    prior = pickled_prior[j]
    PPL_context = np.load(base_path+'/E. semantic_features/PPL_context_%s_%s.npy' % (model_ids[2],story_id))
    # performance
    match_path = 'C:/Users/hchen/Dropbox/PycharmProjects/false_mem/GPT_output/GPT_matching/%s' % story_id
    SAVE_PATH = base_path+'/GPT_output/GPT_summary'
    confab = np.load(SAVE_PATH+'/confab_%s_%s.npy'%(story_id,subfolder))[:, keep_ids[j]]
    true = np.load(SAVE_PATH+'/conflict_%s_%s.npy'%(story_id,subfolder))[:, keep_ids[j]]
    inference = np.load(SAVE_PATH+'/inference_%s_%s.npy'%(story_id,subfolder))[:, keep_ids[j]]
    correct = vectorize_correct(true, confab).astype(float)
    false = vectorize_false(true,confab).astype(float)
    # add data
    for i in range(confab.shape[0]):  # person
        for k in range(confab.shape[1]):  # event
            if np.isnan(confab[i,k]):
                recall = 0
            else:
                recall = 1
            par_id = os.path.basename(recall_paths[i]).split('_')[0]
            event_id = find_index_for_sum(keep_ids[j], k+1)
            df_all.loc[n] = [story_id, par_id, event_id, centrality[k], PPL_context[k], prior[k],
                             recall, true[i,k], confab[i,k], correct[i,k],false[i,k],
                             inference[i,k],len(story_segs[k])]
            n+=1

df_all.to_excel('./B. script_analysis/3. semantic_analysis\data_sheets/semantic_measures_A.xlsx')
