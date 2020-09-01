import statistics
import json
import pickle

if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    all_sims = json.load(open(pkl_dump_dir + "all_sims_label_top_words_labels.json", "r"))

    std_dev = {}
    for lbl in all_sims:
        vals = list(all_sims[lbl].values())
        std_dev[lbl] = statistics.stdev(vals)

    json.dump(std_dev, open(pkl_dump_dir + "std_dev_sim_label_top_words_labels.json", "w"))
