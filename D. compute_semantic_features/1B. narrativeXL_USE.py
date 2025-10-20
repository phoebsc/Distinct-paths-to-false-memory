"""
from the NarrativeXL dataset, extract an average embedding as the general "prior knowledge"
"""
import pandas as pd
import os, re, pickle
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

WINDOW_SIZE = 55
col_sep = "###THISISOURUNIQUECOLUMNSEPARATORUNLIKELYTOBEENCOUNTEREDINANYBOOK###"
line_sep = "###THISISOURUNIQUELINESEPARATORUNLIKELYTOBEENCOUNTEREDINANYBOOK###"
basepath = 'C:/Users/hchen/Dropbox/PycharmProjects/false_mem/B. script_analysis/helper_files/'
# TODO change these paths
path1 = os.path.join(basepath, '../helper_files/NarrativeXL_DATA_Oct21_2023/TrueAndFalseSummaries/TestSet500')
path2 = os.path.join(basepath, '../helper_files/NarrativeXL_DATA_Oct21_2023/TrueAndFalseSummaries/TrainSet1000')
save_path = os.path.join(basepath, '../helper_files/narrativeXL_embed/USE/')

files_test = os.listdir(path1)
files_train = os.listdir(path2)
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
def embed_USE(input_texts):
    embedding = embed(input_texts)  # input must be a list of strings
    embedding_numpy = embedding.numpy()
    return embedding_numpy

def read_file(filename):
    with open(filename, "r",errors="ignore") as f:
        content = f.read()
    # Split the content by the custom line separator to get rows
    raw_rows = content.split(line_sep)
    # Build a list of rows (lists of columns)
    all_rows = []
    for row in raw_rows:
        # Skip entirely blank lines (if any appear)
        if not row.strip():
            continue
        columns = row.split(col_sep)
        all_rows.append(columns)
    # The first row is presumably your header
    header = all_rows[0]
    data_rows = all_rows[1:]
    # Create a DataFrame
    df = pd.DataFrame(data_rows, columns=header)
    txt = ' '.join(df['BookChunks'])
    del df
    return txt

if __name__ == "__main__":
    # test files
    for ff in files_test:
        filename = os.path.join(path1, ff)
        if os.path.isfile(save_path + os.path.basename(filename).split('.')[0] + '_embed.pkl'):
            print('skip ' + os.path.basename(filename))
            continue
        txt = read_file(filename)
        txt = txt.replace('\\n', ' ')
        txt = txt.replace('\\t', ' ')
        txt = txt.replace('\\', ' ')
        txt = txt.replace('\n', ' ')
        txt = re.sub(' +', ' ', txt)
        txt = txt.split()
        txt = [' '.join(txt[i:i + WINDOW_SIZE]) for i in range(0, len(txt), WINDOW_SIZE)]

        embeddings = embed_USE(txt)

        with open(os.path.join(save_path, os.path.basename(filename).split('.')[0] + '_embed.pkl'), 'wb') as f:
            pickle.dump(embeddings, f)
    # training files
    for ff in files_train:
        filename = os.path.join(path2, ff)
        if os.path.isfile(save_path + os.path.basename(filename).split('.')[0] + '_embed.pkl'):
            print('skip ' + os.path.basename(filename))
            continue
        txt = read_file(filename)
        txt = txt.replace('\\n', ' ')
        txt = txt.replace('\\t', ' ')
        txt = txt.replace('\\', ' ')
        txt = txt.replace('\n', ' ')
        txt = re.sub(' +', ' ', txt)
        txt = txt.split()
        txt = [' '.join(txt[i:i + WINDOW_SIZE]) for i in range(0, len(txt), WINDOW_SIZE)]

        embeddings = embed_USE(txt)

        with open(os.path.join(save_path, os.path.basename(filename).split('.')[0] + '_embed.pkl'), 'wb') as f:
            pickle.dump(embeddings, f)
