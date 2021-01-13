import pickle
from sklearn.metrics import classification_report


def measure_quality(df_fine, temp_df, l):
    true = []
    preds = [l] * len(temp_df)
    for i, row in temp_df.iterrows():
        inds = df_fine[df_fine["text"] == row["text"]].index.values
        true.append(df_fine["label"][inds[0]])
    print(classification_report(true, preds), flush=True)


def func(df_fine, df, p="sports"):
    parent_df = df[df["label"].isin([p])].reset_index(drop=True)

    print("Tennis")
    reg_exp = "|".join(["golf", "soccer", "basketball", "hockey", "baseball", "football"])
    child_df = parent_df[parent_df.text.str.contains("tennis") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "tennis")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "tennis" + ".pkl", "wb"))

    print("Golf")
    reg_exp = "|".join(
        ["tennis", "soccer", "basketball", "hockey", "baseball", "football"])
    child_df = parent_df[parent_df.text.str.contains("golf") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "golf")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "golf" + ".pkl", "wb"))

    print("Soccer")
    reg_exp = "|".join(["tennis", "golf", "basketball", "hockey", "baseball", "football"])
    child_df = parent_df[
        parent_df.text.str.contains("soccer") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "soccer")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "soccer" + ".pkl", "wb"))

    print("Basketball")
    reg_exp = "|".join(["tennis", "golf", "soccer", "hockey", "baseball", "football"])
    child_df = parent_df[
        parent_df.text.str.contains("basketball") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "basketball")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "basketball" + ".pkl", "wb"))

    print("Hockey")
    reg_exp = "|".join(["tennis", "golf", "soccer", "basketball", "baseball", "football"])
    child_df = parent_df[
        parent_df.text.str.contains("hockey") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "hockey")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "hockey" + ".pkl", "wb"))

    print("Baseball")
    reg_exp = "|".join(["tennis", "golf", "soccer", "basketball", "hockey", "football"])
    child_df = parent_df[
        parent_df.text.str.contains("baseball") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "baseball")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "baseball" + ".pkl", "wb"))

    print("Football")
    reg_exp = "|".join(["tennis", "golf", "soccer", "basketball", "hockey", "baseball"])
    child_df = parent_df[
        parent_df.text.str.contains("football") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "football")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "football" + ".pkl", "wb"))


if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_coarse.pkl", "rb"))
    df_fine = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    func(df_fine, df)
