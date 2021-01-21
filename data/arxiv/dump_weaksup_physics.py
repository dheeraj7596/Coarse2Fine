import pickle
from sklearn.metrics import classification_report


def measure_quality(df_fine, temp_df, l):
    true = []
    preds = [l] * len(temp_df)
    for i, row in temp_df.iterrows():
        inds = df_fine[df_fine["text"] == row["text"]].index.values
        true.append(df_fine["label"][inds[0]])
    print(classification_report(true, preds), flush=True)


def func(df_fine, df, p="physics"):
    parent_df = df[df["label"].isin([p])].reset_index(drop=True)
    all_words = {"optics", "fluid dynamics", "atomic", "instrumentation", "detectors", "accelerator physics",
                 "plasma", "chemical", "society"}

    print("physics.optics")
    reg_exp = "|".join(list(all_words - {"optics"}))
    child_df = parent_df[
        parent_df.text.str.contains("optics") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "physics.optics")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "physics.optics" + ".pkl", "wb"))

    print("physics.flu-dyn")
    reg_exp = "|".join(list(all_words - {"fluid dynamics"}))
    child_df = parent_df[
        parent_df.text.str.contains("fluid dynamics") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "physics.flu-dyn")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "physics.flu-dyn" + ".pkl", "wb"))

    print("physics.atom-ph")
    reg_exp = "|".join(list(all_words - {"atomic"}))
    child_df = parent_df[
        parent_df.text.str.contains("atomic") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "physics.atom-ph")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "physics.atom-ph" + ".pkl", "wb"))

    print("physics.ins-det")
    reg_exp = "|".join(list(all_words - {"instrumentation", "detectors"}))
    child_df = parent_df[
        parent_df.text.str.contains("instrumentation|detectors") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "physics.ins-det")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "physics.ins-det" + ".pkl", "wb"))

    print("physics.acc-ph")
    reg_exp = "|".join(list(all_words - {"accelerator"}))
    child_df = parent_df[
        parent_df.text.str.contains("accelerator") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "physics.acc-ph")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "physics.acc-ph" + ".pkl", "wb"))

    print("physics.plasm-ph")
    reg_exp = "|".join(list(all_words - {"plasma"}))
    child_df = parent_df[
        parent_df.text.str.contains("plasma") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "physics.plasm-ph")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "physics.plasm-ph" + ".pkl", "wb"))

    print("physics.chem-ph")
    reg_exp = "|".join(list(all_words - {"chemical"}))
    child_df = parent_df[
        parent_df.text.str.contains("chemical") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "physics.chem-ph")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "physics.chem-ph" + ".pkl", "wb"))

    print("physics.soc-ph")
    reg_exp = "|".join(list(all_words - {"society"}))
    child_df = parent_df[
        parent_df.text.str.contains("society") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "physics.soc-ph")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "physics.soc-ph" + ".pkl", "wb"))


if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "arxiv/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_coarse.pkl", "rb"))
    df_fine = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    func(df_fine, df)
