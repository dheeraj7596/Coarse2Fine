import pickle
from classifier_seedwords import preprocess_df
from util import fit_get_tokenizer
from nltk.corpus import stopwords
from get_seed_words import decipher_phrase
from get_skip_grams import encode_phrase
import json


def get_conditional_probability(texts, a, b, mode="doc"):
    # computes p(b|a)
    num = 0
    den = 0
    if mode == "doc":
        for sent in texts:
            tokens = set(sent.strip().split())
            if a in tokens:
                den += 1
                if b in tokens:
                    num += 1
        if den != 0:
            return num / den
        else:
            return 0


if __name__ == "__main__":
    # base_path = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    base_path = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt"
    data_path = base_path + dataset + "/"

    df = pickle.load(open(data_path + "df_coarse_phrase.pkl", "rb"))
    phrase_id = pickle.load(open(data_path + "phrase_id_coarse_map.pkl", "rb"))
    id_phrase_map = pickle.load(open(data_path + "id_phrase_coarse_map.pkl", "rb"))
    df = preprocess_df(df)
    tokenizer = fit_get_tokenizer(df.text, max_words=150000)
    # child_to_parent = pickle.load(open(data_path + "child_to_parent.pkl", "rb"))
    parent_to_child = pickle.load(open(data_path + "parent_to_child.pkl", "rb"))

    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    words = {}
    for p in ["sports", "arts", "science"]:
        temp_df = df[df.label.isin([p])].reset_index(drop=True)
        tokenizer = fit_get_tokenizer(temp_df.text, max_words=150000)
        for ch in parent_to_child[p]:
            words[ch] = {}
            child_label_str = encode_phrase(" ".join([t for t in ch.split("_") if t not in stop_words]).strip(),
                                            phrase_id)
            thresh = get_conditional_probability(temp_df.text, child_label_str, encode_phrase(p, phrase_id))
            print("Threshold for ", p, ch, str(thresh))
            for tok in tokenizer.word_index:
                prob = get_conditional_probability(temp_df.text, tok, child_label_str)
                if prob >= thresh:
                    words[ch][decipher_phrase(tok, id_phrase_map)] = prob

    json.dump(words, open(data_path + "conditional_prob_doc.json", "w"))
