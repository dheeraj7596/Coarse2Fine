import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"

    pkl_dump_dir = basepath + dataset
    plot_dump_dir = pkl_dump_dir + "plots/"
    all_sims = json.load(open(pkl_dump_dir + "all_sims_label_top_words_labels.json", "r"))

    for l in all_sims:
        values = list(all_sims[l].values())
        plt.figure()
        n = plt.hist(values, color='blue', edgecolor='black', bins=100)
        plt.savefig(plot_dump_dir + l + ".png")
