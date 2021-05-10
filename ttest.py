import pickle
import numpy as np
from sklearn.metrics import f1_score

if __name__ == "__main__":
    data_path = "./data/20news/"
    probs_df = pickle.load(open(data_path + "probs_df.pkl", "rb"))

    perf1 = []
    perf2 = []

    for i in range(100):
        print(i)
        temp_df = probs_df.sample(frac=0.3).reset_index(drop=True)
        perf1.append(f1_score(temp_df["true"], temp_df["clf1"], average='macro'))
        perf2.append(f1_score(temp_df["true"], temp_df["clf2"], average='macro'))

    f1 = open(data_path + "res1.txt", "w")
    f2 = open(data_path + "res2.txt", "w")

    for c in perf1:
        f1.write(str(c))
        f1.write("\n")

    for c in perf2:
        f2.write(str(c))
        f2.write("\n")

    f1.close()
    f2.close()
