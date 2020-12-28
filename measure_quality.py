import pickle
from sklearn.metrics import classification_report

if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))
    fine_labels = set(df.label.values)

    for l in fine_labels:
        print("Label", l)
        temp_df = pickle.load(open(pkl_dump_dir + "exclusive/" + l + ".pkl", "rb"))
        true = []
        preds = [l] * len(temp_df)
        for i, row in temp_df.iterrows():
            inds = df[df["text"] == row["text"]].index.values
            true.append(df["label"][inds[0]])
        print(classification_report(true, preds), flush=True)
