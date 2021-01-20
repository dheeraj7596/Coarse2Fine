import pickle
from sklearn.metrics import classification_report


def measure_quality(df_fine, temp_df, l):
    true = []
    preds = [l] * len(temp_df)
    for i, row in temp_df.iterrows():
        inds = df_fine[df_fine["text"] == row["text"]].index.values
        true.append(df_fine["label"][inds[0]])
    print(classification_report(true, preds), flush=True)


def func(df_fine, df, p="cs"):
    parent_df = df[df["label"].isin([p])].reset_index(drop=True)
    all_words = {"computer vision", "game theory", "information theory", "artificial intelligence",
                 "distributed computing", "parallel computing", "cluster computing", "natural language processing",
                 "networking", "internet architecture", "software engineering", "complexity classes",
                 "cryptography", "security", "machine learning", "modal logic", "control system", "society",
                 "data structures", "programming languages", "database management"}

    print("cs.CV")
    reg_exp = "|".join(list(all_words - {"computer vision"}))
    child_df = parent_df[
        parent_df.text.str.contains("computer vision") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "cs.CV")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "cs.CV" + ".pkl", "wb"))

    print("cs.GT")
    reg_exp = "|".join(list(all_words - {"game theory"}))
    child_df = parent_df[
        parent_df.text.str.contains("game theory") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "cs.GT")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "cs.GT" + ".pkl", "wb"))

    print("cs.IT")
    reg_exp = "|".join(list(all_words - {"information theory"}))
    child_df = parent_df[
        parent_df.text.str.contains("information theory") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "cs.IT")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "cs.IT" + ".pkl", "wb"))

    print("cs.AI")
    reg_exp = "|".join(list(all_words - {"artificial intelligence"}))
    child_df = parent_df[
        parent_df.text.str.contains("artificial intelligence") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "cs.AI")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "cs.AI" + ".pkl", "wb"))

    print("cs.DC")
    reg_exp = "|".join(list(all_words - {"distributed computing", "parallel computing"}))
    child_df = parent_df[
        parent_df.text.str.contains("distributed computing|parallel computing") & ~parent_df.text.str.contains(
            reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "cs.DC")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "cs.DC" + ".pkl", "wb"))

    print("cs.CL")
    reg_exp = "|".join(list(all_words - {"natural language processing"}))
    child_df = parent_df[
        parent_df.text.str.contains("natural language processing") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "cs.CL")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "cs.CL" + ".pkl", "wb"))

    print("cs.NI")
    reg_exp = "|".join(list(all_words - {"networking", "internet architecture"}))
    child_df = parent_df[
        parent_df.text.str.contains("networking|internet architecture") & ~parent_df.text.str.contains(
            reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "cs.NI")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "cs.NI" + ".pkl", "wb"))

    print("cs.SE")
    reg_exp = "|".join(list(all_words - {"software engineering"}))
    child_df = parent_df[
        parent_df.text.str.contains("software engineering") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "cs.SE")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "cs.SE" + ".pkl", "wb"))

    print("cs.CC")
    reg_exp = "|".join(list(all_words - {"complexity classes"}))
    child_df = parent_df[
        parent_df.text.str.contains("complexity classes") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "cs.CC")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "cs.CC" + ".pkl", "wb"))

    print("cs.CR")
    reg_exp = "|".join(list(all_words - {"cryptography", "security"}))
    child_df = parent_df[
        parent_df.text.str.contains("cryptography|security") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "cs.CR")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "cs.CR" + ".pkl", "wb"))

    print("cs.LG")
    reg_exp = "|".join(list(all_words - {"machine learning"}))
    child_df = parent_df[
        parent_df.text.str.contains("machine learning") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "cs.LG")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "cs.LG" + ".pkl", "wb"))

    print("cs.LO")
    reg_exp = "|".join(list(all_words - {"modal logic"}))
    child_df = parent_df[
        parent_df.text.str.contains("modal logic") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "cs.LO")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "cs.LO" + ".pkl", "wb"))

    print("cs.SY")
    reg_exp = "|".join(list(all_words - {"control system"}))
    child_df = parent_df[
        parent_df.text.str.contains("control system") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "cs.SY")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "cs.SY" + ".pkl", "wb"))

    print("cs.CY")
    reg_exp = "|".join(list(all_words - {"society"}))
    child_df = parent_df[
        parent_df.text.str.contains("society") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "cs.CY")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "cs.CY" + ".pkl", "wb"))

    print("cs.DS")
    reg_exp = "|".join(list(all_words - {"data structures"}))
    child_df = parent_df[
        parent_df.text.str.contains("data structures") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "cs.DS")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "cs.DS" + ".pkl", "wb"))

    print("cs.PL")
    reg_exp = "|".join(list(all_words - {"programming languages"}))
    child_df = parent_df[
        parent_df.text.str.contains("programming languages") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "cs.PL")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "cs.PL" + ".pkl", "wb"))

    print("cs.DB")
    reg_exp = "|".join(list(all_words - {"database management"}))
    child_df = parent_df[
        parent_df.text.str.contains("database management") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "cs.DB")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "cs.DB" + ".pkl", "wb"))


if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "arxiv/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_coarse.pkl", "rb"))
    df_fine = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    func(df_fine, df)
