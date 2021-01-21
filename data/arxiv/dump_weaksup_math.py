import pickle
from sklearn.metrics import classification_report


def measure_quality(df_fine, temp_df, l):
    true = []
    preds = [l] * len(temp_df)
    for i, row in temp_df.iterrows():
        inds = df_fine[df_fine["text"] == row["text"]].index.values
        true.append(df_fine["label"][inds[0]])
    print(classification_report(true, preds), flush=True)


def func(df_fine, df, p="math"):
    parent_df = df[df["label"].isin([p])].reset_index(drop=True)
    all_words = {"numerical analysis", "algebraic geometry", "banach spaces", "number theory",
                 "complex variables", "boundary conditions", "optimal control", "statistical inference",
                 "central limit theorems",
                 "differential geometry", "combinatorics", "von neumann algebras", "representation theory",
                 "orthogonal polynomials", "dynamical systems", "finite groups", "quantum algebra", "set theory",
                 "rings",
                 "symplectic geometry", "algebraic topology", "commutative algebra", "geometric topology",
                 "metric geometry"}

    print("math.NA")
    reg_exp = "|".join(list(all_words - {"numerical analysis"}))
    child_df = parent_df[
        parent_df.text.str.contains("numerical analysis") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.NA")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.NA" + ".pkl", "wb"))

    print("math.AG")
    reg_exp = "|".join(list(all_words - {"algebraic geometry"}))
    child_df = parent_df[
        parent_df.text.str.contains("algebraic geometry") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.AG")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.AG" + ".pkl", "wb"))

    print("math.FA")
    reg_exp = "|".join(list(all_words - {"banach spaces"}))
    child_df = parent_df[
        parent_df.text.str.contains("banach spaces") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.FA")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.FA" + ".pkl", "wb"))

    print("math.NT")
    reg_exp = "|".join(list(all_words - {"number theory"}))
    child_df = parent_df[
        parent_df.text.str.contains("number theory") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.NT")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.NT" + ".pkl", "wb"))

    print("math.CV")
    reg_exp = "|".join(list(all_words - {"complex variables"}))
    child_df = parent_df[
        parent_df.text.str.contains("complex variables") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.CV")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.CV" + ".pkl", "wb"))

    print("math.AP")
    reg_exp = "|".join(list(all_words - {"boundary conditions"}))
    child_df = parent_df[
        parent_df.text.str.contains("boundary conditions") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.AP")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.AP" + ".pkl", "wb"))

    print("math.OC")
    reg_exp = "|".join(list(all_words - {"optimal control"}))
    child_df = parent_df[
        parent_df.text.str.contains("optimal control") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.OC")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.OC" + ".pkl", "wb"))

    print("math.ST")
    reg_exp = "|".join(list(all_words - {"statistical inference"}))
    child_df = parent_df[
        parent_df.text.str.contains("statistical inference") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.ST")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.ST" + ".pkl", "wb"))

    print("math.PR")
    reg_exp = "|".join(list(all_words - {"central limit theorems"}))
    child_df = parent_df[
        parent_df.text.str.contains("central limit theorems") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.PR")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.PR" + ".pkl", "wb"))

    print("math.DG")
    reg_exp = "|".join(list(all_words - {"differential geometry"}))
    child_df = parent_df[
        parent_df.text.str.contains("differential geometry") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.DG")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.DG" + ".pkl", "wb"))

    print("math.CO")
    reg_exp = "|".join(list(all_words - {"combinatorics"}))
    child_df = parent_df[
        parent_df.text.str.contains("combinatorics") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.CO")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.CO" + ".pkl", "wb"))

    print("math.OA")
    reg_exp = "|".join(list(all_words - {"von neumann algebras"}))
    child_df = parent_df[
        parent_df.text.str.contains("von neumann algebras") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.OA")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.OA" + ".pkl", "wb"))

    print("math.RT")
    reg_exp = "|".join(list(all_words - {"representation theory"}))
    child_df = parent_df[
        parent_df.text.str.contains("representation theory") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.RT")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.RT" + ".pkl", "wb"))

    print("math.CA")
    reg_exp = "|".join(list(all_words - {"orthogonal polynomials"}))
    child_df = parent_df[
        parent_df.text.str.contains("orthogonal polynomials") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.CA")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.CA" + ".pkl", "wb"))

    print("math.DS")
    reg_exp = "|".join(list(all_words - {"dynamical systems"}))
    child_df = parent_df[
        parent_df.text.str.contains("dynamical systems") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.DS")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.DS" + ".pkl", "wb"))

    print("math.GR")
    reg_exp = "|".join(list(all_words - {"finite groups"}))
    child_df = parent_df[
        parent_df.text.str.contains("finite groups") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.GR")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.GR" + ".pkl", "wb"))

    print("math.QA")
    reg_exp = "|".join(list(all_words - {"quantum algebra"}))
    child_df = parent_df[
        parent_df.text.str.contains("quantum algebra") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.QA")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.QA" + ".pkl", "wb"))

    print("math.LO")
    reg_exp = "|".join(list(all_words - {"set theory"}))
    child_df = parent_df[
        parent_df.text.str.contains("set theory") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.LO")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.LO" + ".pkl", "wb"))

    print("math.RA")
    reg_exp = "|".join(list(all_words - {"rings"}))
    child_df = parent_df[
        parent_df.text.str.contains("rings") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.RA")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.RA" + ".pkl", "wb"))

    print("math.SG")
    reg_exp = "|".join(list(all_words - {"symplectic geometry"}))
    child_df = parent_df[
        parent_df.text.str.contains("symplectic geometry") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.SG")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.SG" + ".pkl", "wb"))

    print("math.AT")
    reg_exp = "|".join(list(all_words - {"algebraic topology"}))
    child_df = parent_df[
        parent_df.text.str.contains("algebraic topology") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.AT")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.AT" + ".pkl", "wb"))

    print("math.AC")
    reg_exp = "|".join(list(all_words - {"commutative algebra"}))
    child_df = parent_df[
        parent_df.text.str.contains("commutative algebra") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.AC")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.AC" + ".pkl", "wb"))

    print("math.GT")
    reg_exp = "|".join(list(all_words - {"geometric topology"}))
    child_df = parent_df[
        parent_df.text.str.contains("geometric topology") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.GT")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.GT" + ".pkl", "wb"))

    print("math.MG")
    reg_exp = "|".join(list(all_words - {"metric geometry"}))
    child_df = parent_df[
        parent_df.text.str.contains("metric geometry") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "math.MG")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "math.MG" + ".pkl", "wb"))


if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "arxiv/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_coarse.pkl", "rb"))
    df_fine = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    func(df_fine, df)
