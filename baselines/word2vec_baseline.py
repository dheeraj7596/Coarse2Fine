import itertools
import os
import pickle
import re
from collections import Counter
from os import path
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import word2vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import json
import sys

sys.path.append("./")
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from gensim.models import word2vec
import pickle
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def build_vocab(sentences):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return word_counts, vocabulary, vocabulary_inv


def get_embeddings(inp_data, vocabulary_inv, size_features=100,
                   mode='skipgram',
                   min_word_count=2,
                   context=5):
    model_name = "embedding"
    model_name = os.path.join(model_name)
    num_workers = 15  # Number of threads to run in parallel
    downsampling = 1e-3  # Downsample setting for frequent words
    print('Training Word2Vec model...')
    sentences = [[vocabulary_inv[w] for w in s] for s in inp_data]
    if mode == 'skipgram':
        sg = 1
        print('Model: skip-gram')
    elif mode == 'cbow':
        sg = 0
        print('Model: CBOW')
    embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                        sg=sg,
                                        size=size_features,
                                        min_count=min_word_count,
                                        window=context,
                                        sample=downsampling)
    embedding_model.init_sims(replace=True)
    print("Saving Word2Vec model {}".format(model_name))
    embedding_weights = np.zeros((len(vocabulary_inv), size_features))
    for i in range(len(vocabulary_inv)):
        word = vocabulary_inv[i]
        if word in embedding_model:
            embedding_weights[i] = embedding_model[word]
        else:
            embedding_weights[i] = np.random.uniform(-0.25, 0.25,
                                                     embedding_model.vector_size)
    return embedding_weights


def get_label_embeddings(labels, label_term_dict):
    label_embeddings = []
    for class_label in labels:
        embedding_list = []
        for word in label_term_dict[class_label]:
            embedding_list.append(embedding_weights[vocabulary[word]])
        label_embeddings.append(np.array(embedding_list).sum(axis=0) / len(label_term_dict[class_label]))
    return label_embeddings


def get_doc_embeddings():
    doc_weights = []
    for doc in inp_data:
        if len(doc) == 0:
            doc_weights.append(np.random.uniform(-0.25, 0.25, 100))
        else:
            doc_weights.append(embedding_weights[doc].sum(axis=0) / len(doc))
    return doc_weights


if __name__ == "__main__":
    basepath = "../data/"
    dataset = "20news/"
    pkl_dump_dir = basepath + dataset

    with open(pkl_dump_dir + "df_fine.pkl", "rb") as handler:
        df = pickle.load(handler)

    print(set(df.label))

    label_term_dict = json.load(open(pkl_dump_dir + "surface_name_mapping.json", "r"))

    # for doc in data:
    #     assert(len(doc) > 0)
    tagged_data = [word_tokenize(_d) for i, _d in enumerate(df["text"])]
    word_counts, vocabulary, vocabulary_inv = build_vocab(tagged_data)
    inp_data = [[vocabulary[word] for word in text] for text in tagged_data]
    embedding_weights = get_embeddings(inp_data, vocabulary_inv)

    label_to_ind = {}
    ind_to_label = {}
    labels = set(df["label"])
    for i, l in enumerate(labels):
        label_to_ind[l] = i
        ind_to_label[i] = l

    label_embeddings = get_label_embeddings(labels, label_term_dict)
    doc_embeddings = get_doc_embeddings()
    similarities = cosine_similarity(doc_embeddings, label_embeddings)
    y_pred = np.argmax(similarities, axis=1)
    # print classification report
    y_true = []
    for l in df["label"]:
        y_true.append(label_to_ind[l])
    print(classification_report(y_true, y_pred))
