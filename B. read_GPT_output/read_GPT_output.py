import os
import pandas as pd
import re
import numpy as np
"""
Script to convert the GPT output from event matching task and memory classification task to arrays for further analysis
"""
story_id = 'baseball'
match_path = './GPT_output/event_matching/%s' % story_id
recall_path = './GPT_output/memory_classification/%s' % story_id  # the already rated recall transcripts
story_segs = pd.read_excel('./A. script_prompting/story_segs.xlsx',
                           sheet_name='%s_segs' % story_id)['events'].to_list()
SAVE_PATH = './GPT_output/results_arrays/'
recall_paths = [recall_path + '/'+x for x in os.listdir(recall_path)]
"""
save confab, factual error and inference results into arrays
"""
confab = np.ones((len(recall_paths), len(story_segs)))
confab = -1 * confab
true = np.ones((len(recall_paths), len(story_segs)))
true = 2 * true
inference = np.ones((len(recall_paths), len(story_segs)))
inference = -1 * inference
unmatch = 0
matched = 0
wrong_perc = []
for p, path in enumerate(recall_paths):
    df = pd.read_excel(path)  # false mem result
    df_match = pd.read_excel(match_path + '/' + os.path.basename(path))  # matching result
    df = df.merge(df_match, on='sentence',how='left')
    wrong_sent = 0
    for i in range(len(df)):
        try:
            pattern1 = r"1\)\s*none"
            pattern2 = r"2\)\s*none"
            pattern3 = r"3\)\s*none"
            # factual error
            match1 = re.search(pattern1,df.loc[i, 'output'].lower())
            if match1:
                truth = 1
            else:
                truth = 0
            # confabulation
            match2 = re.search(pattern2,df.loc[i, 'output'].lower())
            if match2:
                con = 0
            else:
                con = 1
            # inference
            match3 = re.search(pattern3, df.loc[i, 'output'].lower())
            if match3:
                infer = 0
            else:
                infer = 1
            event_id = int(df.loc[i, 'ind'])
            # if event_id is zero, then there is no match
            if event_id == 0:
                unmatch += 1
                continue
            # counting sentences
            matched += 1
            if con==1 or truth==0:
                wrong_sent += 1
            # if already has a recalled event
            confab[p, event_id - 1] = max(con, confab[p, event_id - 1])
            true[p, event_id - 1] = min(truth, true[p, event_id - 1])
            inference[p, event_id - 1] = max(infer, inference[p, event_id - 1])
        except:
            unmatch += 1
            print(path, i)
    wrong_perc.append(wrong_sent/len(df))
# cleaning
confab[confab<0] = np.nan
inference[inference<0] = np.nan
true[true>1] = np.nan
np.save(SAVE_PATH+'/confab_%s'% story_id,confab)
np.save(SAVE_PATH+'/conflict_%s'% story_id,true)
np.save(SAVE_PATH+'/inference_%s'% story_id,inference)

"""
inspection
"""
# inspection
events_insp = np.arange(1, len(story_segs) + 1)
df_result = pd.DataFrame(columns=['subject','event_id','sentence', 'output'])
for p, path in enumerate(recall_paths):
    df = pd.read_excel(path)
    df_match = pd.read_excel(match_path + '/' + os.path.basename(path))  # matching result
    df = df.merge(df_match, on='sentence',how='left')
    df = df[~df['output'].isnull()]
    df.index = np.arange(len(df))
    # pool results
    df_add = pd.DataFrame(columns=['subject','event_id','sentence', 'output','true','con'])
    k = 0
    for i in range(len(df)):
        try:
            pattern1 = r"1\)\s*none"
            pattern2 = r"2\)\s*none"
            match1 = re.search(pattern1, df.loc[i, 'output'].lower())
            if match1:
                truth = 1
            else:
                truth = 0
            match2 = re.search(pattern2, df.loc[i, 'output'].lower())
            if match2:
                con = 0
            else:
                con = 1
            event_id = int(df.loc[i, 'ind'])
            # if event_id is zero, then there is no match
            if event_id == 0:
                continue
            df_add.loc[k] = [path,event_id, df.loc[i, 'sentence'],  df.loc[i, 'output'],truth,con]
            k+=1
        except:
            pass
    df_result = pd.concat([df_result, df_add])
df_result.to_excel('./GPT_output/GPT_summary/inspect_%s_%s.xlsx' % (story_id, subfolder))

