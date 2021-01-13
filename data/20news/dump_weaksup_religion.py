import pickle
from sklearn.metrics import classification_report


def measure_quality(df_fine, temp_df, l):
    true = []
    preds = [l] * len(temp_df)
    for i, row in temp_df.iterrows():
        inds = df_fine[df_fine["text"] == row["text"]].index.values
        true.append(df_fine["label"][inds[0]])
    print(classification_report(true, preds), flush=True)


def func(df_fine, df, p="religion"):
    parent_df = df[df["label"].isin([p])].reset_index(drop=True)

    print("alt.atheism")
    reg_exp = "|".join(["christian"])
    child_df = parent_df[
        parent_df.text.str.contains("atheism") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "alt.atheism")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "alt.atheism" + ".pkl", "wb"))

    print("soc.religion.christian")
    reg_exp = "|".join(["atheism"])
    child_df = parent_df[
        parent_df.text.str.contains("christian") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "soc.religion.christian")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "soc.religion.christian" + ".pkl", "wb"))


if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "20news/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_coarse.pkl", "rb"))
    df_fine = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    func(df_fine, df)
