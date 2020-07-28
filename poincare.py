import pickle
from gensim.models.poincare import PoincareModel
from nltk.corpus import stopwords

if __name__ == "__main__":
    # base_path = "/data4/dheeraj/coarse2fine/"
    base_path = "./data/"
    dataset = "nyt"
    data_path = base_path + dataset + "/"

    df = pickle.load(open(data_path + "df_coarse.pkl", "rb"))
    child_to_parent = pickle.load(open(data_path + "child_to_parent.pkl", "rb"))
    parent_to_child = pickle.load(open(data_path + "parent_to_child.pkl", "rb"))
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')

    relations = set()
    for child in child_to_parent:
        relations.add((child, child_to_parent[child]))

    for i, row in df.iterrows():
        label = row["label"]
        sent = row["text"]
        words_list = sent.strip().split()
        filtered_words = [word for word in words_list if word not in stop_words]
        child_labels = parent_to_child[label]
        for child_label in child_labels:
            temp_word = " ".join(child_label.split("_"))
            if temp_word in sent:
                for w in filtered_words:
                    relations.add((w, child_label))

    print("Number of relations: ", len(relations))
    model = PoincareModel(list(relations))
    model.train(epochs=1, print_every=500)
    pass
