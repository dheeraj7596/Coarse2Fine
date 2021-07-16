import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sys
import pandas as pd
import os
import numpy as np


def get_high_quality_inds(true, preds, pred_probs, label_to_index, num, threshold=0.7):
    pred_inds = []
    for p in preds:
        pred_inds.append(label_to_index[p])

    pred_label_to_inds = {}
    for i, p in enumerate(pred_inds):
        try:
            pred_label_to_inds[p].append(i)
        except:
            pred_label_to_inds[p] = [i]

    label_to_probs = {}
    min_ct = float("inf")
    for p in pred_label_to_inds:
        label_to_probs[p] = []
        ct_thresh = 0
        for ind in pred_label_to_inds[p]:
            temp = pred_probs[ind][p]
            if temp >= threshold:
                ct_thresh += 1
            label_to_probs[p].append(temp)
        min_ct = min(min_ct, ct_thresh)
    # min_ct = min(min_ct, int((percent_threshold / (len(label_to_index) * 100.0)) * len(preds)))
    min_ct = num
    print("Collecting", min_ct, "samples as high quality")
    final_inds = {}
    for p in label_to_probs:
        probs = label_to_probs[p]
        inds = np.array(probs).argsort()[-min_ct:][::-1]
        final_inds[p] = []
        for i in inds:
            final_inds[p].append(pred_label_to_inds[p][i])

    temp_true = []
    temp_preds = []
    for p in final_inds:
        for ind in final_inds[p]:
            temp_true.append(true[ind])
            temp_preds.append(preds[ind])

    print("Classification Report of High Quality data")
    print(classification_report(temp_true, temp_preds), flush=True)
    return final_inds


if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    basepath = "/data/dheeraj/coarse2fine/"
    dataset = sys.argv[4] + "/"
    # dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    iteration = int(sys.argv[1])
    # iteration = 1

    p = sys.argv[2]
    # p = "science"
    dump_flag = sys.argv[3]
    # dump_flag = 0
    algo = sys.argv[5]
    # algo = "ce_hinge"

    # use_gpu = False
    if sys.argv[6] == "nyt":
        num_dic = {"arts": 46, "science": 21, "politics": 24, "sports": 270, "business": 33}
    elif sys.argv[6] == "20news":
        num_dic = {"science": 112, "recreation": 69, "computer": 65, "religion": 110, "politics": 24}
    elif sys.argv[6] == "arxiv":
        num_dic = {"cs": 56, "math": 43, "physics": 74}
    else:
        raise Exception("Unknown label detected")

    df_train = pickle.load(open(pkl_dump_dir + algo + "/df_gen_" + p + ".pkl", "rb"))
    print(df_train["text"][0], df_train["label"][0], flush=True)
    df_fine = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))
    df_test = df_fine[df_fine["label"].isin(list(set(df_train.label.values)))].reset_index(drop=True)

    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))

    for ch in parent_to_child[p]:
        child_df = pickle.load(
            open(pkl_dump_dir + "exclusive/" + algo + "/" + str(iteration) + "it/" + ch + ".pkl", "rb"))
        for i in range(1, iteration + 1):
            temp_child_df = pickle.load(
                open(pkl_dump_dir + "exclusive/" + algo + "/" + str(i) + "it/" + ch + ".pkl", "rb"))
            if i == 1:
                child_df = temp_child_df
            else:
                child_df = pd.concat([child_df, temp_child_df])
        child_df["label"] = [ch] * len(child_df)
        df_train = pd.concat([df_train, child_df])

    print(df_train.label.value_counts())

    label_set = set(df_train.label.values)
    label_to_index = {}
    index_to_label = {}
    for i, l in enumerate(list(label_set)):
        label_to_index[l] = i
        index_to_label[i] = l

    vectorizer = TfidfVectorizer(stop_words="english")
    clf = LogisticRegression()

    X_train = vectorizer.fit_transform(df_train["text"])
    X_test = vectorizer.transform(df_test["text"])
    y_train = [label_to_index[l] for l in df_train["label"]]
    y_test = [label_to_index[l] for l in df_test["label"]]

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    preds = [index_to_label[l] for l in preds]
    pred_probs = clf.predict_proba(X_test)

    y_test = [index_to_label[l] for l in y_test]
    high_quality_inds = get_high_quality_inds(y_test, preds, pred_probs, label_to_index, num_dic[p], threshold=0.7)

    if dump_flag:
        for p in high_quality_inds:
            inds = high_quality_inds[p]
            temp_df = df_test.loc[inds].reset_index(drop=True)
            os.makedirs(pkl_dump_dir + "exclusive/" + algo + "/" + str(iteration + 1) + "it", exist_ok=True)
            pickle.dump(temp_df, open(
                pkl_dump_dir + "exclusive/" + algo + "/" + str(iteration + 1) + "it/" + index_to_label[p] + ".pkl",
                "wb"))
