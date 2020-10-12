import json
from label_cluster import generate_pseudo_labels, modify_seeds
import pickle
from util import fit_get_tokenizer
from sklearn.metrics import classification_report

if __name__ == "__main__":
    base_path = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # base_path = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt"
    data_path = base_path + dataset + "/"

    df = pickle.load(open(data_path + "df_fine_phrase.pkl", "rb"))
    phrase_id = pickle.load(open(data_path + "phrase_id_coarse_map.pkl", "rb"))
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
    for p in parent_labels:
        children = parent_to_child[p]
        children_label_term_dict = {}
        for c in children:
            children_label_term_dict[c] = label_term_dict[c]

        temp_df = df[df.label.isin(children)].reset_index(drop=True)
        print("Length of df for parent", p, len(temp_df))
        tokenizer = fit_get_tokenizer(temp_df.text, max_words=150000)
        X, y, y_true = generate_pseudo_labels(temp_df, children, children_label_term_dict, tokenizer)
        X_all += X
        y_all += y
        y_true_all += y_true

    print(classification_report(y_true_all, y_all))
