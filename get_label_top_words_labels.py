import pickle
from nltk.corpus import stopwords
from util import cosine_similarity
import string
import json

if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"

    pkl_dump_dir = basepath + dataset
    all_sims = {}
    mean_sim = {}

    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    embeddings = pickle.load(open(pkl_dump_dir + "bert_word_phrase_embeddings.pkl", "rb"))
    label_embeddings = pickle.load(open(pkl_dump_dir + "label_embeddings.pkl", "rb"))

    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    for p in parent_to_child:
        for ch in parent_to_child[p]:
            all_sims[ch] = {}
            mean_sim[ch] = 0
            for w in embeddings:
                child_label_str = " ".join([t for t in ch.split("_") if t not in stop_words]).strip()
                sim = cosine_similarity(label_embeddings[child_label_str], embeddings[w])
                all_sims[ch][w] = sim
                mean_sim[ch] += sim
            mean_sim[ch] = mean_sim[ch] / len(embeddings)
            all_sims[ch] = {k: v for k, v in sorted(all_sims[ch].items(), key=lambda item: -item[1])[:1000]}

    print(mean_sim)
    json.dump(all_sims, open(pkl_dump_dir + "all_sims_label_top_words_labels.json", "w"))
    json.dump(mean_sim, open(pkl_dump_dir + "mean_sim_label_top_words_labels.json", "w"))
