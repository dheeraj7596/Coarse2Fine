import pickle
from sklearn.metrics import classification_report


def measure_quality(df_fine, temp_df, l):
    true = []
    preds = [l] * len(temp_df)
    for i, row in temp_df.iterrows():
        inds = df_fine[df_fine["text"] == row["text"]].index.values
        true.append(df_fine["label"][inds[0]])
    print(classification_report(true, preds), flush=True)


def func(df_fine, df, p="recreation"):
    parent_df = df[df["label"].isin([p])].reset_index(drop=True)

    print("rec.autos")
    reg_exp = "|".join(["motorcycles", "motorcycle", "baseball", "hockey"])
    child_df = parent_df[
        parent_df.text.str.contains("autos|automobiles|automobile") & ~parent_df.text.str.contains(
            reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "rec.autos")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "rec.autos" + ".pkl", "wb"))

    print("rec.motorcycles")
    reg_exp = "|".join(["autos", "automobile", "automobiles", "baseball", "hockey"])
    child_df = parent_df[
        parent_df.text.str.contains("motorcycles|motorcycle") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "rec.motorcycles")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "rec.motorcycles" + ".pkl", "wb"))

    print("rec.sport.baseball")
    reg_exp = "|".join(["autos", "automobile", "automobiles", "motorcycles", "motorcycle", "hockey"])
    child_df = parent_df[parent_df.text.str.contains("baseball") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "rec.sport.baseball")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "rec.sport.baseball" + ".pkl", "wb"))

    print("rec.sport.hockey")
    reg_exp = "|".join(["autos", "automobile", "automobiles", "motorcycles", "motorcycle", "baseball"])
    child_df = parent_df[
        parent_df.text.str.contains("hockey") & ~parent_df.text.str.contains(
            reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "rec.sport.hockey")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "rec.sport.hockey" + ".pkl", "wb"))


if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "20news/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_coarse.pkl", "rb"))
    df_fine = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    func(df_fine, df)
