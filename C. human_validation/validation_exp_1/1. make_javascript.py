import pandas as pd
import re, json
import numpy as np

story_ids = ['pieman', 'eyespy', 'oregontrail', 'baseball']
txt_for_irb = []
for story_id in story_ids:
    df_trials = pd.read_excel('./D. human_validation/validation_exp_March2025/trial_excels/%s_agreement.xlsx' % story_id)
    df_trials = df_trials.sort_values(by='event_id', ascending=True)
    # remove attention trials
    df_trials = df_trials.loc[df_trials['subject'] != 'attention']
    df_trials['event_id'] = df_trials['event_id'].astype(int)
    story_segs = pd.read_excel(
        'C:/Users/hchen\Dropbox\school_projects/2023/firebase_experiments\causality_rating\materials/story_segs.xlsx',
        sheet_name='%s_segs' % story_id)['events'].to_list()
    if story_id in ['eyespy', 'oregontrail']:
        file = open("./D. human_validation/validation_exp_Sept2024/trial_excels/%s_punc.txt" % story_id, "r")
        content = file.read()
        story_segs = content.split('\n\n')
    stim_list = []
    for block_n in np.arange(0,len(df_trials),5):
        survey_trials = []
        st, end = block_n, block_n + 5
        for i in np.arange(st, end):
            id = df_trials.iloc[i]['ID']
            story_event_n_minus_1 = story_segs[df_trials.iloc[i]['event_id']-2]
            story_event_n = story_segs[df_trials.iloc[i]['event_id']-1]
            try:
                story_event_n_plus_1 = story_segs[df_trials.iloc[i]['event_id']]
            except:
                story_event_n_plus_1 = ''
            recollection = df_trials.iloc[i]['sentence']
            if df_trials.iloc[i]['Question']:
                rater_feedback = "The rater thinks the recollection contains inaccuracies."
                output = df_trials.iloc[i]['output'].split("3)")[0]
                output = output.replace('\'','')
                recollection = recollection.replace('\'','')
                underlined = re.findall(r'"(.*?)"', output)

            else:
                rater_feedback = "The rater thinks the recollection is accurate."
                underlined = ['']
            txt_for_irb.append(recollection)
            survey_trials.append({'story_event_n_minus_1': story_event_n_minus_1,
                              'recollection': recollection,
                              'story_event_n': story_event_n,
                              'story_event_n_plus_1': story_event_n_plus_1,
                              'rater_feedback': rater_feedback,
                              'underlined':underlined,
                              'ID':int(id),
                              'event_id': int(df_trials.iloc[i]['event_id']),
                                'context':'<p><br></p>'.join([story_event_n_minus_1,story_event_n,story_event_n_plus_1])})
        stim_list.append(survey_trials)
    string_json = json.dumps(stim_list)
    print(string_json)

print('\n\n'.join(txt_for_irb))

"""
type of error
"""
df_all = pd.DataFrame()
for story_id in story_ids:
    df_trials = pd.read_excel('./D. human_validation/validation_exp_March2025/trial_excels/%s_agreement.xlsx' % story_id)
    df_all = pd.concat([df_all,df_trials])
df_all['type of error'].value_counts()