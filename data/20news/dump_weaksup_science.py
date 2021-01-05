import pickle
from sklearn.metrics import classification_report


def measure_quality(df_fine, temp_df, l):
    true = []
    preds = [l] * len(temp_df)
    for i, row in temp_df.iterrows():
        inds = df_fine[df_fine["text"] == row["text"]].index.values
        true.append(df_fine["label"][inds[0]])
    print(classification_report(true, preds), flush=True)


def func(df_fine, df, p="science"):
    parent_df = df[df["label"].isin([p])].reset_index(drop=True)

    print("sci.crypt")
    reg_exp = "|".join(["electronics", "medicine", "space"])
    child_df = parent_df[
        parent_df.text.str.contains("encryption") & ~parent_df.text.str.contains(
            reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "sci.crypt")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "sci.crypt" + ".pkl", "wb"))

    print("sci.electronics")
    reg_exp = "|".join(["encryption", "medicine", "space"])
    child_df = parent_df[
        parent_df.text.str.contains("electronics") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "sci.electronics")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "sci.electronics" + ".pkl", "wb"))

    print("sci.med")
    reg_exp = "|".join(["encryption", "electronics", "space"])
    child_df = parent_df[parent_df.text.str.contains("medicine") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "sci.med")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "sci.med" + ".pkl", "wb"))

    print("sci.space")
    reg_exp = "|".join(["encryption", "electronics", "medicine"])
    child_df = parent_df[
        parent_df.text.str.contains("space") & ~parent_df.text.str.contains(
            reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "sci.space")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "sci.space" + ".pkl", "wb"))


if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "20news/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_coarse.pkl", "rb"))
    df_fine = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    func(df_fine, df)
