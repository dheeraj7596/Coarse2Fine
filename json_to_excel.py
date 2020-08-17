import json
import pandas as pd

if __name__ == "__main__":
    base_path = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    dataset = "nyt"
    data_path = base_path + dataset + "/"

    seeds = json.load(open(data_path + "conwea_top100words_phrases.json", "r"))

    for lbl in seeds:
        words = list(seeds[lbl].keys())
        df = pd.DataFrame.from_dict({"words": words})
        df.to_excel(data_path + lbl + "words_phrases.xlsx")
