import os
import pandas as pd
import tiktoken
from langchain.schema import (
    SystemMessage
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

def clean_text(text):
    unwanted_char = '\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x7f\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff'
    text = "".join([(" " if n in unwanted_char else n) for n in text if n not in unwanted_char])
    return text

def encoding_getter(encoding_type: str):
    """
    Returns the appropriate encoding based on the given encoding type (either an encoding string or a model name).
    """
    if "k_base" in encoding_type:
        return tiktoken.get_encoding(encoding_type)
    else:
        return tiktoken.encoding_for_model(encoding_type)

def tokenizer(string: str, encoding_type: str) -> list:
    """
    Returns the tokens in a text string using the specified encoding.
    """
    encoding = encoding_getter(encoding_type)
    tokens = encoding.encode(string)
    return tokens

os.environ["OPENAI_API_KEY"] = 'INSERT_KEY'
model = 'gpt-4o'
TRANSCRIPT_PATH = 'INSERT_DATASET_PATH'
for story_id in ['pieman','eyespy','oregontrail','baseball']:
    chat = ChatOpenAI(model_name=model,temperature=0)
    # system message
    with open('./A. script_prompting/memory_classification_prompts/%s_prompt.txt' % story_id,
              encoding='utf-8') as f:
        content = f.read()
        system, examples = content.split('##')
    system_message_prompt = SystemMessage(content=system)
    # human message template
    human_template = "Current conversation: {history} \nPsychologist: {input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    # few shot examples
    # examples = [e for e in examples.split('\nuser message: ') if len(e)>1]
    examples = [e for e in examples.split('\nPsychologist: ') if len(e)>1]
    example_prompts = []
    for example in examples:
        human, ai = example.split('\nanswer: ')
        human = HumanMessagePromptTemplate.from_template('The next recall sentence:'+human)
        ai = AIMessagePromptTemplate.from_template(ai.replace('\n',''))
        example_prompts.extend([human,ai])
    # putting them together
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt] + example_prompts + [human_message_prompt])

    """
    prepare input and run
    """
    transcript_folder = os.path.join(TRANSCRIPT_PATH,story_id)
    files = os.listdir(transcript_folder)
    for file in files:
        file_path = './GPT_output/GPT_rating/%s_prompt%s/FINISH/' % story_id + file+'.xlsx'
        if os.path.isfile(file_path):
            continue
        # split
        with open(os.path.join(transcript_folder,file),encoding='latin1') as f:
            contents = clean_text(f.read())
        sents = contents.replace('?','.').replace('!','.').split('.')
        # if fewer than 10 words, combine with the previous sentence
        new_sents = []
        for sent in sents:
            nletter = len([s for s in sent.split(' ') if len(s)>1])
            if nletter < 10 and len(new_sents) > 0:  # if shorter than 10 words, combine with the previous sentence
                new_sents[-1] = new_sents[-1]+' '+sent+'.'
            else:
                new_sents.append(sent+'.')
        """
        run and save result
        """
        conversation = ConversationChain(
            prompt=chat_prompt,
            llm=chat,
            verbose=True,
            memory=ConversationBufferWindowMemory(k=5)
        )
        df = pd.DataFrame(columns=['sentence', 'output', 'note'])
        n = 0
        for sent in new_sents:
            output = conversation.predict(input='The next recall sentence: """'+sent+'"""').replace('\nAI: ','').replace('AI: ','')
            note = ''
            df.loc[n] = [sent, output, note]
            n += 1
        df.to_excel('./GPT_output/memory_classification/' % (story_id) + file+'.xlsx')
