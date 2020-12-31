import pickle
from sklearn.metrics import classification_report


def measure_quality(df_fine, temp_df, l):
    true = []
    preds = [l] * len(temp_df)
    for i, row in temp_df.iterrows():
        inds = df_fine[df_fine["text"] == row["text"]].index.values
        true.append(df_fine["label"][inds[0]])
    print(classification_report(true, preds), flush=True)


def func(df_fine, df, p="arts"):
    parent_df = df[df["label"].isin([p])].reset_index(drop=True)

    print("dance")
    reg_exp = "|".join(["music", "movies", "movie", "television", "tv"])
    child_df = parent_df[
        parent_df.text.str.contains("dance") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "dance")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "dance" + ".pkl", "wb"))

    print("music")
    reg_exp = "|".join(["dance", "movies", "movie", "television", "tv"])
    child_df = parent_df[
        parent_df.text.str.contains("music") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "music")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "music" + ".pkl", "wb"))

    print("movies")
    reg_exp = "|".join(["dance", "music", "television", "tv"])
    child_df = parent_df[
        parent_df.text.str.contains("movies") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "movies")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "movies" + ".pkl", "wb"))

    print("television")
    reg_exp = "|".join(["dance", "movies", "movie", "music"])
    child_df = parent_df[
        parent_df.text.str.contains("television") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "television")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "television" + ".pkl", "wb"))


if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_coarse.pkl", "rb"))
    df_fine = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    func(df_fine, df)
