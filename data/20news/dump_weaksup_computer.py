import pickle
from sklearn.metrics import classification_report


def measure_quality(df_fine, temp_df, l):
    true = []
    preds = [l] * len(temp_df)
    for i, row in temp_df.iterrows():
        inds = df_fine[df_fine["text"] == row["text"]].index.values
        true.append(df_fine["label"][inds[0]])
    print(classification_report(true, preds), flush=True)


def func(df_fine, df, p="computer"):
    parent_df = df[df["label"].isin([p])].reset_index(drop=True)

    print("comp.graphics")
    reg_exp = "|".join(["windows", "ibm", "mac", "x window", "xterm"])
    child_df = parent_df[
        parent_df.text.str.contains("graphics") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "comp.graphics")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "comp.graphics" + ".pkl", "wb"))

    print("comp.os.ms-windows.misc")
    reg_exp = "|".join(["graphics", "ibm", "mac", "x window", "xterm"])
    child_df = parent_df[parent_df.text.str.contains("windows") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "comp.os.ms-windows.misc")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "comp.os.ms-windows.misc" + ".pkl", "wb"))

    print("comp.sys.ibm.pc.hardware")
    reg_exp = "|".join(["graphics", "windows", "mac", "x window", "xterm"])
    child_df = parent_df[parent_df.text.str.contains("ibm") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "comp.sys.ibm.pc.hardware")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "comp.sys.ibm.pc.hardware" + ".pkl", "wb"))

    print("comp.sys.mac.hardware")
    reg_exp = "|".join(["graphics", "windows", "ibm", "x window", "xterm"])
    child_df = parent_df[
        parent_df.text.str.contains("mac") & ~parent_df.text.str.contains(
            reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "comp.sys.mac.hardware")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "comp.sys.mac.hardware" + ".pkl", "wb"))

    print("comp.windows.x")
    reg_exp = "|".join(["graphics", "ibm", "mac", "windows"])
    child_df = parent_df[parent_df.text.str.contains("x window|xterm") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "comp.windows.x")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "comp.windows.x" + ".pkl", "wb"))


if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "20news/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_coarse.pkl", "rb"))
    df_fine = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    func(df_fine, df)
