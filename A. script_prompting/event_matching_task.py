import os, re
import pandas as pd
from langchain.schema import (
    SystemMessage
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

from langchain import LLMChain
os.environ["OPENAI_API_KEY"] = "INSERT_KEY"

for story_id in ['pieman','eyespy','oregontrail','baseball']:
    match_path = './GPT_output/event_matching/%s' % story_id
    recall_path = './GPT_output/memory_classification/%s' % story_id  # the already rated recall transcripts
    story_segs = pd.read_excel('./A. script_prompting/story_segs.xlsx',
                               sheet_name='%s_segs' % story_id)['events'].to_list()
    chat = ChatOpenAI(model_name='gpt-4o', temperature=0.3)

    # system message
    with open('./A. script_prompting/event_matching_prompts/%s_matchprompt.txt' % story_id, encoding='windows-1252') as f:
        content = f.read()
        system = content
    system_message_prompt = SystemMessage(content=system)
    # human message template
    human_template = "Current conversation: {history} \nHuman: {input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # putting them together
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    """
    prepare input and run
    """
    def clean_text(text):
        unwanted_char = '\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x7f\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff'
        text = "".join([(" " if n in unwanted_char else n) for n in text if n not in unwanted_char])
        return text

    recall_paths = [os.path.join(recall_path,x)
                    for x in os.listdir(recall_path) if '.xlsx' in x]
    for path in recall_paths:
        df = pd.read_excel(path)
        sents = df['sentence'].to_list()
        conversation = ConversationChain(
            prompt=chat_prompt,
            llm=chat,
            verbose=False,
            memory=ConversationBufferWindowMemory(k=5)
        )
        """
        run and save result
        """
        answers = []
        indices = []
        segs = []
        for sent in sents:
            output = conversation.predict(input='"""'+sent+'"""').replace('\nAI: ','').replace('AI: ','')
            answers.append(output)
            try:
                ind = int(re.findall(r'\b\d+\b', output)[0])
                indices.append(ind)
                segs.append(story_segs[ind-1])
            except:
                indices.append('')
                segs.append('')
        df['answer'] = answers
        df['ind'] = indices
        df['seg'] = segs
        df_match = df[['sentence', 'ind', 'seg','answer']]
        df_match.to_excel(match_path+'/'+os.path.basename(path))
