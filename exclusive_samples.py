import pickle
from classifier_seedwords import preprocess_df
from util import fit_get_tokenizer
from nltk.corpus import stopwords
from get_seed_words import decipher_phrase
from get_skip_grams import encode_phrase
import json, math


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


def get_conditional_probability_words(texts, tokenizer, mode="doc"):
    # computes p(b|a)
    num = {}
    den = {}
    prob = {}
    if mode == "doc":
        for sent in texts:
            tokens = set(sent.strip().split())
            for tok in tokens:
                try:
                    den[tok] += 1
                except:
                    den[tok] = 1
                if child_label_str in tokens:
                    try:
                        num[tok] += 1
                    except:
                        num[tok] = 1

        for tok in tokenizer.word_index:
            assert tok in den
            try:
                if num[tok] == 0 or den[tok] == 0:
                    prob[tok] = 0
                else:
                    prob[tok] = num[tok] / den[tok]
            except:
                prob[tok] = 0
    return prob


def get_pmi(texts, a, b, mode="doc"):
    con_num = 0
    con_den = 0
    den = 0
    if mode == "doc":
        for sent in texts:
            tokens = set(sent.strip().split())
            if a in tokens:
                con_den += 1
                if b in tokens:
                    con_num += 1
            if b in tokens:
                den += 1

        if con_den == 0 or den == 0 or con_num == 0:
            return -math.inf
        else:
            return math.log((con_num * len(texts)) / (con_den * den))


if __name__ == "__main__":
    base_path = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # base_path = "/data4/dheeraj/coarse2fine/"
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
    threshold = {}
    probability = {}
    for p in ["sports", "arts", "science"]:
        temp_df = df[df.label.isin([p])].reset_index(drop=True)
        tokenizer = fit_get_tokenizer(temp_df.text, max_words=150000)
        for ch in parent_to_child[p]:
            words[ch] = {}
            child_label_str = encode_phrase(" ".join([t for t in ch.split("_") if t not in stop_words]).strip(),
                                            phrase_id)
            thresh = get_conditional_probability(temp_df.text, child_label_str, encode_phrase(p, phrase_id))
            # thresh = get_pmi(temp_df.text, child_label_str, encode_phrase(p, phrase_id))
            print("Threshold for ", p, ch, str(thresh))
            prob = get_conditional_probability_words(temp_df.text, tokenizer)
            # prob = get_pmi(temp_df.text, tok, child_label_str)
            for tok in tokenizer.word_index:
                assert tok in prob
                if prob[tok] >= thresh:
                    words[ch][decipher_phrase(tok, id_phrase_map)] = prob[tok]
            threshold[ch] = thresh
            probability[ch] = prob

        for ch in parent_to_child[p]:
            siblings = set(parent_to_child[p]) - {ch}
            removed_words = []
            for word in words[ch]:
                for sb in siblings:
                    try:
                        if probability[sb][word] >= words[ch][word] or probability[sb][word] >= threshold[sb]:
                            removed_words.append(word)
                    except:
                        continue

            for w in removed_words:
                words[ch].pop(w, None)

    # json.dump(words, open(data_path + "conditional_prob_doc.json", "w"))
    json.dump(words, open(data_path + "pmi_doc.json", "w"))
