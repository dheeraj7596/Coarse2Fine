import pickle
from util import fit_get_tokenizer
from get_skip_grams import encode_phrase
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import math
from scipy.special import softmax
from sklearn.metrics import classification_report
import numpy as np


def generate_pseudo_labels_pmi(df, labels, tokenizer, probability, phrase_id):
    def argmax_label(count_dict):
        maxi = -math.inf
        max_label = None
        for l in count_dict:
            if count_dict[l] > maxi:
                maxi = count_dict[l]
                max_label = l
        return max_label

    def compute_ln(df, labels, phrase_id, index_word):
        lns = []
        stemmer = SnowballStemmer("english")
        stop_words = set(stopwords.words('english'))
        mod_labels = []
        for l in labels:
            lns.append(0)
            label_str = encode_phrase(" ".join([stemmer.stem(t) for t in l.split("_") if
                                                stemmer.stem(t) not in stop_words and t not in stop_words]).strip(),
                                      phrase_id)
            mod_labels.append(label_str)
        for i, row in df.iterrows():
            line = row["text"]
            tokens = tokenizer.texts_to_sequences([line])[0]
            words = []
            for tok in tokens:
                words.append(index_word[tok])
            temp = list(set(words).intersection(mod_labels))
            if len(temp) == 1:
                lns[mod_labels.index(temp[0])] += 1

        probs = [i / sum(lns) for i in lns]
        ln_probs = [math.log(p) for p in probs]
        ln_dict = {}
        for i, l in enumerate(labels):
            ln_dict[l] = ln_probs[i]
        return ln_dict

    y = []
    X = []
    y_true = []
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w
    label_ln = compute_ln(df, labels, phrase_id, index_word)

    for index, row in df.iterrows():
        line = row["text"]
        label = row["label"]
        tokens = tokenizer.texts_to_sequences([line])[0]
        words = []
        for tok in tokens:
            words.append(index_word[tok])
        count_dict = {}
        for l in labels:
            count_dict[l] = label_ln[l]
            for word in words:
                try:
                    if probability[l][word] == -math.inf:
                        continue
                    count_dict[l] += probability[l][word]
                except:
                    continue
        lbl = argmax_label(count_dict)
        y.append(lbl)
        X.append(line)
        y_true.append(label)
    return X, y, y_true


if __name__ == "__main__":
    base_path = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # base_path = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt"
    data_path = base_path + dataset + "/"
    probability = pickle.load(open(data_path + "label_pmi_map.pkl", "rb"))

    df = pickle.load(open(data_path + "df_fine_phrase_stem.pkl", "rb"))
    phrase_id = pickle.load(open(data_path + "phrase_id_coarse_stem_map.pkl", "rb"))
    id_phrase_map = pickle.load(open(data_path + "id_phrase_coarse_stem_map.pkl", "rb"))

    child_to_parent = pickle.load(open(data_path + "child_to_parent.pkl", "rb"))
    parent_to_child = pickle.load(open(data_path + "parent_to_child.pkl", "rb"))

    parent_labels = ["sports", "arts"]

    X_all = []
    y_all = []
    y_true_all = []
    words_common_all = []
    for p in parent_labels:
        children = parent_to_child[p]

        temp_df = df[df.label.isin(children)].reset_index(drop=True)
        print("Length of df for parent", p, len(temp_df))
        tokenizer = fit_get_tokenizer(temp_df.text, max_words=150000)
        X, y, y_true = generate_pseudo_labels_pmi(temp_df, children, tokenizer, probability, phrase_id)
        X_all += X
        y_all += y
        y_true_all += y_true

    print(classification_report(y_true_all, y_all))
