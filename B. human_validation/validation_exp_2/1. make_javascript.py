import pandas as pd
import re, json, random
import numpy as np


###
def generate_QA(event_df):
    # Compute some metric from sub_df, which is a DataFrame of each subgroup
    event_df_copy = event_df.copy().loc[~event_df.Question.isnull()]
    if len(event_df_copy) == 0:
        return []
    else:
        questions = []
        for i in range(len(event_df_copy)):
            prompt = event_df_copy.iloc[i]['Question']
            name = int(event_df_copy.iloc[i]['ID'])  # ID column
            options = [x.lstrip() for x in event_df_copy.iloc[i]['Options'].split('/')]
            random.shuffle(options)
            questions.append({'prompt': clean_smart_quotes(prompt),
                             'name': 'excel_id'+str(name),
                             'options': options, 'required':True})
            # print the attention trials
            if event_df_copy.iloc[i]['subject']=='attention':
                print(clean_smart_quotes(prompt),
                        '\n', options,'\n\n')
        return questions

def clean_smart_quotes(text):
    return text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").replace("–", "-")
##
story_ids = ['pieman', 'eyespy', 'oregontrail', 'baseball']

for story_id in story_ids:
    df_trials = pd.read_excel('./D. human_validation/validation_exp_feb/trial_excels/%s_agreement.xlsx' % story_id)
    df_trials = df_trials.sort_values(by='event_id', ascending=True)
    story_segs = pd.read_excel(
        'C:/Users/hchen\Dropbox\school_projects/2023/firebase_experiments\causality_rating\materials/story_segs.xlsx',
        sheet_name='%s_segs' % story_id)['events'].to_list()
    if story_id in ['eyespy', 'oregontrail']:
        file = open("./D. human_validation/validation_exp_sept/trial_excels/%s_punc.txt" % story_id, "r")
        content = file.read()
        story_segs = content.split('\n\n')
    stim_list = []
    # new stim
    result = df_trials.groupby('event_id').apply(generate_QA)
    for i in range(len(result)):
        if result.iloc[i] == []:
            continue
        id = int(list(result.index)[i])
        story_event_n_minus_1 = story_segs[id - 2]
        story_event_n = story_segs[id - 1]
        try:
            story_event_n_plus_1 = story_segs[id]
        except:
            story_event_n_plus_1 = ''
        context = story_event_n_minus_1+'\n\n'+story_event_n+'\n\n'+story_event_n_plus_1
        # also add the entire story before and after
        context_before = '\n\n'.join(story_segs[0:id-2])
        context_after = '\n\n'.join(story_segs[id:])
        stim_list.append({'context_current': context,
                          'context_before': context_before,
                          'context_after': context_after,
                         'questions': result.iloc[i]})
    string_json = json.dumps(stim_list, ensure_ascii=False)
    print(string_json)

"""
for IRB
"""
df_all = pd.DataFrame(columns=['Question','Options'])
for story_id in story_ids:
    df_trials = pd.read_excel('./D. human_validation/validation_exp_feb/trial_excels/%s_agreement.xlsx' % story_id)
    df_trials = df_trials.sort_values(by='event_id', ascending=True)
    df_all = pd.concat([df_all,df_trials[['Question','Options']]])
df_all = df_all.dropna()
for i in range(len(df_all)):
    print(str(i+1)+'.',df_all.iloc[i]['Question'])
    print(df_all.iloc[i]['Options'])