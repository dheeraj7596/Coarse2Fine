import json
from label_cluster import generate_pseudo_labels, modify_seeds
import pickle
from util import fit_get_tokenizer, plot_confusion_mat
import pandas as pd
from sklearn.metrics import classification_report
from get_seed_words import decipher_phrase
import matplotlib.pyplot as plt

if __name__ == "__main__":
    base_path = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # base_path = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt"
    data_path = base_path + dataset + "/"
    dump_csv = False

    df = pickle.load(open(data_path + "df_fine_phrase.pkl", "rb"))
    phrase_id = pickle.load(open(data_path + "phrase_id_coarse_map.pkl", "rb"))
    id_phrase_map = pickle.load(open(data_path + "id_phrase_coarse_map.pkl", "rb"))
    # child_seeds_dict = json.load(open(data_path + "conditional_prob_doc_all_filters.json", "r"))
    child_seeds_dict = json.load(open(data_path + "pmi_doc_all_filters.json", "r"))
    child_to_parent = pickle.load(open(data_path + "child_to_parent.pkl", "rb"))
    parent_to_child = pickle.load(open(data_path + "parent_to_child.pkl", "rb"))
    child_labels = list(child_seeds_dict.keys())
    parent_labels = set([child_to_parent[ch] for ch in child_labels])

    label_term_dict = modify_seeds(child_seeds_dict, phrase_id)

    X_all = []
    y_all = []
    y_true_all = []
    words_common_all = []
    for p in parent_labels:
        children = parent_to_child[p]
        children_label_term_dict = {}
        for c in children:
            children_label_term_dict[c] = label_term_dict[c]

        temp_df = df[df.label.isin(children)].reset_index(drop=True)
        print("Length of df for parent", p, len(temp_df))
        tokenizer = fit_get_tokenizer(temp_df.text, max_words=150000)
        X, y, y_true, words_cmn = generate_pseudo_labels(temp_df, children, children_label_term_dict, tokenizer)
        X_all += X
        y_all += y
        y_true_all += y_true
        words_common_all += words_cmn

    for i, s in enumerate(words_common_all):
        words_common_all[i] = " ".join([decipher_phrase(w, id_phrase_map) for w in s.strip().split()])

    if dump_csv:
        df_tmp = pd.DataFrame.from_dict(
            {"text": X_all, "pseudo_label": y_all, "true_label": y_true_all, "words_cmn": words_common_all})
        df_tmp.to_csv(data_path + "pseudo.csv")
    print(classification_report(y_true_all, y_all))

    plot_confusion_mat(y_true_all, y_all, list(set(y_true_all)))
    plt.show()
