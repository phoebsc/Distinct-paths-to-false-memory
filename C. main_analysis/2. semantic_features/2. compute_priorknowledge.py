"""
from helper_files/NarrativeXL_embed, compute cosime similarity from story events to the narrativeXL embeddings,
and then average cosine similarities per event
- 1500 files including both training and test sets
- all texts were segmented using a 55 word window
- on average, each file has 1536.4 windows. there are in total 2,304,582 windows = 126,752,010 words
- embedding model is all-mpnet-base-v or USE
"""
import pandas as pd
import numpy as np
import pickle, os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub

model_id = 'USE'
if model_id == 'mpnet':
    model = SentenceTransformer('all-mpnet-base-v2')
elif model_id == 'USE':
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
def embed_USE(input_texts):
    embedding = model(input_texts)  # input must be a list of strings
    embedding_numpy = embedding.numpy()
    return embedding_numpy
story_ids = ['pieman','eyespy','oregontrail','baseball']
embed_path = './B. script_analysis/helper_files/NarrativeXL_embed/%s' % model_id
embed_files = [os.path.join(embed_path,x) for x in os.listdir(embed_path)]
keep_ids = np.load('./E. semantic_features/keep_ids.npy', allow_pickle=True)


# compute distance from events to the embeddings, and then average cosine sim
story_sims = []
for j, story_id in enumerate(story_ids):
    story_segs = pd.read_excel('./A. script_prompting/story_segs.xlsx',
                               sheet_name='%s_segs' % story_id)['events'].to_list()
    story_segs = np.array(story_segs)[keep_ids[j]]
    if model_id == 'USE':
        embeddings = embed_USE(story_segs)
    else:
        embeddings = model.encode(story_segs)
    # load narrativeXL
    count = []
    running_sum = []  # 1500 files x story_segs
    for embed_file in embed_files:
        try:
            with open(embed_file, 'rb') as file:
                data = pickle.load(file)
        except:
            print(embed_file)
        sim_array = cosine_similarity(embeddings, data)  # story_segs x current narrative XL file length
        count.append(data.shape[0])
        running_sum.append(np.sum(sim_array, axis=1))
    story_sims.append(np.sum(running_sum,axis=0)/np.sum(count))

with open('./E. semantic_features/PriorKnowledge_allstories_%s.pkl' % model_id, 'wb') as file:
    pickle.dump(story_sims, file)
