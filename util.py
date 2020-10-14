import numpy as np
from scipy import spatial
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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


def fit_get_tokenizer(data, max_words):
    tokenizer = Tokenizer(num_words=max_words, filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(data)
    return tokenizer


def plot_confusion_mat(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    display_labels = labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    return disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation="horizontal", values_format=None)
