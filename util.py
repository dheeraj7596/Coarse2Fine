import numpy as np
from scipy import spatial


def cosine_similarity(a, b):
    return 1 - spatial.distance.cosine(a, b)


def calculate_df_doc_freq(df):
    docfreq = {}
    docfreq["UNK"] = len(df)
    for index, row in df.iterrows():
        line = row["text"]
        words = line.strip().split()
        temp_set = set(words)
        for w in temp_set:
            try:
                docfreq[w] += 1
            except:
                docfreq[w] = 1
    return docfreq


def calculate_doc_freq(docs):
    docfreq = {}
    for doc in docs:
        words = doc.strip().split()
        temp_set = set(words)
        for w in temp_set:
            try:
                docfreq[w] += 1
            except:
                docfreq[w] = 1
    return docfreq


def calculate_inv_doc_freq(df, docfreq):
    inv_docfreq = {}
    N = len(df)
    for word in docfreq:
        inv_docfreq[word] = np.log(N / docfreq[word])
    return inv_docfreq


def get_label_docs_dict(df):
    label_docs_dict = {}
    for index, row in df.iterrows():
        line = row["text"]
        label = row["label"]
        try:
            label_docs_dict[label].append(line)
        except:
            label_docs_dict[label] = [line]
    return label_docs_dict


def print_seed_dict(label_seed_dict):
    for label in label_seed_dict:
        print(label)
        print("*" * 80)
        for val in label_seed_dict[label]:
            print(val, label_seed_dict[label][val])
