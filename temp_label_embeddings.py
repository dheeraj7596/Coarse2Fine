import pickle
from nltk.corpus import stopwords
import string

if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    label_specific_embeddings = pickle.load(open(pkl_dump_dir + "label_bert_word_phrase_embeddings.pkl", "rb"))
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))

    label_embeddings = {}
    for p in parent_to_child:
        for ch in parent_to_child[p]:
            child_label_str = " ".join([t for t in ch.split("_") if t not in stop_words]).strip()
            label_embeddings[ch] = label_specific_embeddings[p][child_label_str]

    pickle.dump(label_embeddings, open(pkl_dump_dir + "label_embeddings.pkl", "wb"))
