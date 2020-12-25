import pickle

if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_coarse.pkl", "rb"))
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    for p in parent_to_child:
        print("Parent label", p)
        parent_df = df[df["label"].isin([p])].reset_index(drop=True)
        children = parent_to_child[p]
        children_strs = []
        str_to_lbl = {}
        for ch in children:
            ch_str = " ".join(ch.strip().split("_"))
            str_to_lbl[ch_str] = ch
            children_strs.append(ch_str)
        for ch_str in children_strs:
            other_labels = list(set(children_strs) - {ch_str})
            reg_exp = "|".join(other_labels)
            child_df = parent_df[
                parent_df.text.str.contains(ch_str) & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
            # child_df = parent_df[parent_df.text.str.contains(ch_str)].reset_index(drop=True)
            print(ch_str, len(child_df))
            pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + str_to_lbl[ch_str] + ".pkl", "wb"))
        print("*" * 80)
