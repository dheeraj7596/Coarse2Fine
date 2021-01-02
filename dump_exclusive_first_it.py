import pickle
import os
from dump_business_weaksup import measure_quality

if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    exclusive_df_dir = pkl_dump_dir + "exclusive/"
    dest_exclusive_df_dir = pkl_dump_dir + "exclusive_1it/"
    os.makedirs(dest_exclusive_df_dir, exist_ok=True)
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    df_fine = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))

    num_dic = {"arts": 46, "science": 21, "politics": 24, "sports": 270, "business": 33}
    for p in ["business"]:
        for ch in parent_to_child[p]:
            print(ch)
            child_df = pickle.load(open(exclusive_df_dir + ch + ".pkl", "rb"))
            if len(child_df) > num_dic[p]:
                child_df = child_df.sample(n=num_dic[p], random_state=42).reset_index(drop=True)
            measure_quality(df_fine, child_df, ch)
            pickle.dump(child_df, open(dest_exclusive_df_dir + ch + ".pkl", "wb"))
