import pickle

if __name__ == "__main__":
    base_path = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    dataset = "nyt"
    data_path = base_path + dataset + "/"

    df = pickle.load(open(data_path + "df_coarse.pkl", "rb"))

    reviews = list(df["text"])

    f = open(data_path + "text.txt", "w")
    for i, r in enumerate(reviews):
        f.write(r)
        f.write("\n")

    f.close()
