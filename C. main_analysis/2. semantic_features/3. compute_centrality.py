import os, re, pickle
import pandas as pd
import numpy as np
from scipy.stats import zscore
import seaborn as sns
from scipy.stats import pearsonr
import torch.nn.functional as F
import tensorflow_hub as hub
from transformers import GPT2Tokenizer, GPT2LMHeadModel
base_path = 'C:/Users\hchen\Dropbox\PycharmProjects/false_mem'
SAVE_PATH = base_path+'/GPT_output/results_arrays/'
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from sentence_transformers import SentenceTransformer


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def compute_cosine_similarity(embeddings):
    norm = np.linalg.norm(embeddings, axis=1)
    norm[norm == 0] = 1
    embeddings_normalized = embeddings / norm[:, np.newaxis]
    return np.dot(embeddings_normalized, embeddings_normalized.T)

def embed_model(embedding_model, txt):
    if embedding_model == 'USE':
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        return embed(txt).numpy()
    elif embedding_model == 'GIST':
        model = SentenceTransformer("avsolatorio/GIST-Embedding-v0")
        return model.encode(txt)
    elif embedding_model == 'mpnet':
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        return model.encode(txt)
    elif embedding_model == 'bge':
        model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        return model.encode(txt)
    elif embedding_model == 'e5':
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
        model = AutoModel.from_pretrained('intfloat/e5-base-v2')
        batch_dict = tokenizer(list(txt), max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        # F.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().numpy()

embedding_model = 'USE'  # replace

"""
compute centrality
"""
keep_ids = np.load(base_path + '/E. semantic_features/keep_ids.npy')
story_ids = ['pieman','eyespy','oregontrail','baseball']

centrality_arr = []
for j, story_id in enumerate(story_ids):
    story_segs = pd.read_excel('./A. script_prompting/story_segs.xlsx',
                               sheet_name='%s_segs' % story_id)['events'].to_list()
    story_segs = np.array(story_segs)[keep_ids[j]]
    embedding = embed_model(embedding_model, story_segs)
    cos_sim = compute_cosine_similarity(embedding)
    n = cos_sim.shape[0]
    # Add nodes with initial centrality values
    np.fill_diagonal(cos_sim, np.nan)
    centrality = np.nanmean(cos_sim, axis=0)

    norm_centrality = zscore(centrality)
    centrality_arr.extend(norm_centrality)

    np.save(base_path + '/E. semantic_features/centrality_%s_%s' % (embedding_model, story_id), centrality)