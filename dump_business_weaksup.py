import pickle
from sklearn.metrics import classification_report


def measure_quality(df_fine, temp_df, l):
    true = []
    preds = [l] * len(temp_df)
    for i, row in temp_df.iterrows():
        inds = df_fine[df_fine["text"] == row["text"]].index.values
        true.append(df_fine["label"][inds[0]])
    print(classification_report(true, preds), flush=True)


def func(df_fine, df, p="business"):
    parent_df = df[df["label"].isin([p])].reset_index(drop=True)

    print("Stocks and Bonds")
    reg_exp = "|".join(["economy", "energy companies", "energy", "international business", "business"])
    child_df = parent_df[
        parent_df.text.str.contains("stocks|bonds") & ~parent_df.text.str.contains(
            reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "stocks_and_bonds")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "stocks_and_bonds" + ".pkl", "wb"))

    print("Economy")
    reg_exp = "|".join(
        ["stocks", "bonds", "energy", "energy companies", "international business", "international"])
    child_df = parent_df[parent_df.text.str.contains("economy") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "economy")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "economy" + ".pkl", "wb"))

    print("Energy Companies")
    reg_exp = "|".join(["stocks", "bonds", "economy", "international business", "international"])
    child_df = parent_df[
        parent_df.text.str.contains("energy") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "energy_companies")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "energy_companies" + ".pkl", "wb"))

    print("International Business")
    reg_exp = "|".join(["stocks", "bonds", "economy", "energy", "energy companies"])
    child_df = parent_df[
        parent_df.text.str.contains("international") & parent_df.text.str.contains(
            "business") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "international_business")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "international_business" + ".pkl", "wb"))


if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_coarse.pkl", "rb"))
    df_fine = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    func(df_fine, df)
