import json

def compute_top_100_meansim(all_sims, child_label):
    return sum(list(all_sims[child_label].values())[:100])/100

if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    all_sims = json.load(open(pkl_dump_dir + "all_sims.json", "r"))
    mean_sim = {}
    for lbl in all_sims:
        mean_sim[lbl] = compute_top_100_meansim(all_sims, lbl)

    json.dump(mean_sim, open(pkl_dump_dir + "mean_sim_top100.json", "w"))