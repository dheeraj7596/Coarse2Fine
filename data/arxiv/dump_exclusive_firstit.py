import pickle
from sklearn.metrics import classification_report
import os


def measure_quality(df_fine, temp_df, l):
    true = []
    preds = [l] * len(temp_df)
    for i, row in temp_df.iterrows():
        inds = df_fine[df_fine["text"] == row["text"]].index.values
        true.append(df_fine["label"][inds[0]])
    print(classification_report(true, preds), flush=True)


if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "arxiv/"
    pkl_dump_dir = basepath + dataset

    exclusive_df_dir = pkl_dump_dir + "exclusive/"
    dest_exclusive_df_dir = pkl_dump_dir + "exclusive_1it/"
    dest_exclusive_df_dir_2 = pkl_dump_dir + "exclusive_ceonly_1it/"
    os.makedirs(dest_exclusive_df_dir, exist_ok=True)
    os.makedirs(dest_exclusive_df_dir_2, exist_ok=True)
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    df_fine = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))

    num_dic = {"cs": 56, "math": 43, "physics": 74}
    for p in num_dic:
        for ch in parent_to_child[p]:
            print(ch)
            child_df = pickle.load(open(exclusive_df_dir + ch + ".pkl", "rb"))
            if len(child_df) > num_dic[p]:
                child_df = child_df.sample(n=num_dic[p], random_state=42).reset_index(drop=True)
            measure_quality(df_fine, child_df, ch)
            pickle.dump(child_df, open(dest_exclusive_df_dir + ch + ".pkl", "wb"))
            pickle.dump(child_df, open(dest_exclusive_df_dir_2 + ch + ".pkl", "wb"))
