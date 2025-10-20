
"""
To estimate GloVe semantic distinctiveness, we obtained a GloVe (Global Vectors for Word Representation; Pennington,
Socher, and Manning (2014)) vector for each of the target words (pre-trained vectors from the Common Crawl corpus,
available at http://nlp.stanford.edu/projects/glove/, version glove.840B.300d.txt), and calculated the mean Cosine
similarity between this vector and all other word vectors in the set.
"""

import os, re, string, math
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
story_ids = ['pieman','eyespy','oregontrail','baseball']
subfolders = ['prompt8','prompt5b','prompt5','prompt1']
glove_path = './B. script_analysis/helper_files/glove.840B.300d.txt'
keep_ids = np.load('./features/keep_ids.npy', allow_pickle=True)


def filter_text(text):
    # If you haven't downloaded these before:
    # nltk.download('punkt')
    # nltk.download('stopwords')

    # 1) Start with default English stopwords
    stop_words = set(stopwords.words('english'))

    # 2) Create your own list of function words / auxiliaries you want to remove
    function_aux_words = {
        "am", "is", "are", "was", "were",
        "have", "has", "had", "having",
        "do", "does", "did", "doing",
        "be", "been", "being",
        "there", "here",
        "he", "she", "it", "they", "we", "you",
        "my", "your", "his", "her", "its", "our", "their",
        "this", "that", "these", "those",
        "i", "me", "him", "her", "us", "them",
        "if", "the", "a", "an", "in", "on", "of", "to", "for",  # etc.
        # add any others you want to remove
    }

    stop_words.update(function_aux_words)
    tokens = word_tokenize(text.lower())

    # Remove stop words and punctuation
    filtered_tokens = [
        word for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]

    return filtered_tokens

wnl = WordNetLemmatizer()

def singular_form(token):
    """
    Utility function to get the noun's singular form using WordNetLemmatizer.
    """
    return wnl.lemmatize(token, pos='n')

def tokenize_and_select(line, filter_tags):
    # 1) Tokenize
    tokens = filter_text(line)

    # 2) Remove punctuation from within each token
    cleaned_tokens = []
    for tok in tokens:
        # Make it lowercase to match typical dictionary keys
        tok_lower = tok.lower()
        # Strip punctuation chars
        tok_clean = "".join(ch for ch in tok_lower if ch not in string.punctuation)
        if tok_clean:  # Skip if empty
            cleaned_tokens.append(tok_clean)

    # 3) POS-tag the cleaned tokens
    tagged = nltk.pos_tag(cleaned_tokens)
    filtered = [
        (word, pos)
        for word, pos in tagged
        if any(pos.startswith(t) for t in filter_tags) and
           (len(word)>2)
    ]
    return filtered
def compute_glove_semantic_distinctiveness(
        glove_path: str,
        word_list: list[str], savepath
) -> dict[str, float]:
    """
    For each word in `word_list`, compute the mean cosine similarity
    between its GloVe vector and all other word vectors in `word_list`.

    Parameters
    ----------
    glove_path : str
        Path to the glove.840B.300d.txt (or similar) GloVe file.
    word_list : list of str
        The words for which you want to compute distinctiveness.
        Typically this is the entire set of n=2,109 words.

    Returns
    -------
    dict of {word : mean_cosine_similarity}
        A dictionary mapping each input word to the average cosine
        similarity between that word's vector and all other vectors
        in the list.
    """

    # Convert word_list to a set for faster membership checking:
    word_set = set(word_list)

    # We will store word->vector in a dictionary first
    word_to_vec = {}

    # Read the GloVe file line by line:
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.rstrip().split(" ")
            w = split_line[0]

            # Only load vectors for words in word_list
            if w in word_set:
                # Convert the 300 numbers to floats
                vec = np.array(split_line[1:], dtype=np.float32)
                word_to_vec[w] = vec

            # Optional optimization: Stop early if we've loaded all words
            if len(word_to_vec) == len(word_list):
                break

    # Confirm that we successfully loaded each word’s vector
    missing_words = [w for w in word_list if w not in word_to_vec]
    if missing_words:
        print(f"Warning: The following words were not found in the GloVe file:\n{missing_words}")

    # Create an ordered list of the words actually found, and stack into a matrix
    # (size: number_of_found_words x 300)
    found_words = list(word_to_vec.keys())
    vectors = np.vstack([word_to_vec[w] for w in found_words])

    # Normalize the vectors to compute cosine similarities via dot product
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero for any zero vector:
    norms[norms == 0] = 1e-9
    normalized_vectors = vectors / norms

    # Compute pairwise cosine similarities:  sim = n_vec A ⋅ n_vec B
    # (where n_vec is a normalized vector)
    # Then for row i, we can compute the average similarity with all other rows.
    similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)

    # Compute mean similarity for each row (excluding self-similarity)
    # We'll subtract 1 for the diagonal (the sim with itself),
    # then divide by (N - 1).
    n = len(found_words)
    sums_of_sims = similarity_matrix.sum(axis=1) - 1.0  # remove self-sim
    means = sums_of_sims / (n - 1)

    # save to df
    df = pd.DataFrame(data=dict(word=found_words, sim=means))
    df.to_excel(savepath)

for j, (story_id, subfolder) in enumerate(zip(story_ids, subfolders)):
    story_segs = pd.read_excel('C:/Users/hchen\Dropbox\school_projects/2023/firebase_experiments\causality_rating\materials/story_segs.xlsx',
                               sheet_name='%s_segs' % story_id)['events'].to_list()
    story_segs = np.array(story_segs)[keep_ids[j]]
    story = ' '.join(story_segs)
    words_to_process = [a for a,b in tokenize_and_select(story, ['NN','VB','JJ','RB'])]
    compute_glove_semantic_distinctiveness(glove_path, words_to_process,
                                           './B. script_analysis/helper_files/glove_measure/glove_%s.xlsx' % story_id)
