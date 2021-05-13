import pickle
import pandas as pd
from sklearn.metrics import f1_score

if __name__ == "__main__":
    data_path = "./data/20news/"
    pred1_path = data_path + "pred1/"
    pred2_path = data_path + "pred2/"

    df_fine = pickle.load(open(data_path + "df_fine.pkl", "rb"))
    parent_to_child = pickle.load(open(data_path + "parent_to_child.pkl", "rb"))

    clf1 = []
    clf2 = []
    true = []

    for p in parent_to_child:
        df_train = pickle.load(open(data_path + "df_gen_" + p + ".pkl", "rb"))
        df_test = df_fine[df_fine["label"].isin(list(set(df_train.label.values)))].reset_index(drop=True)
        pred1 = pickle.load(open(pred1_path + p + ".pkl", "rb"))
        pred2 = pickle.load(open(pred2_path + p + ".pkl", "rb"))
        clf1 += pred1
        clf2 += pred2
        true += list(df_test["label"])
        print(p)
        print(f1_score(true, clf1, average='macro'))
        print(f1_score(true, clf2, average='macro'))

    probs_df = pd.DataFrame.from_dict({"clf1": clf1, "clf2": clf2, "true": true})
    pickle.dump(probs_df, open(data_path + "probs_df.pkl", "wb"))
