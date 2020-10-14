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


def get_conditional_probability_words(texts, label_str, tokenizer, mode="doc"):
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
                if label_str in tokens:
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
    # computes log(P(b|a)/P(b))
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


def get_pmi_words(texts, label_str, tokenizer, mode="doc"):
    # computes log(P(label_str|tok)/P(label_str))
    con_num = {}
    con_den = {}
    den = {}
    prob = {}
    if mode == "doc":
        for sent in texts:
            tokens = set(sent.strip().split())
            for tok in tokens:
                try:
                    con_den[tok] += 1
                except:
                    con_den[tok] = 1
                if label_str in tokens:
                    try:
                        con_num[tok] += 1
                    except:
                        con_num[tok] = 1
            if label_str in tokens:
                try:
                    den[label_str] += 1
                except:
                    den[label_str] = 1

        for tok in tokenizer.word_index:
            try:
                if con_num[tok] == 0 or con_den[tok] == 0 or den[label_str] == 0:
                    prob[tok] = -math.inf
                else:
                    prob[tok] = math.log((con_num[tok] * len(texts)) / (con_den[tok] * den[label_str]))
            except:
                prob[tok] = -math.inf
    return prob


if __name__ == "__main__":
    base_path = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # base_path = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt"
    data_path = base_path + dataset + "/"
    func = "pmi"
    assert func in ["pmi", "cond_prob"]  # this could be either pmi or cond_prob

    df = pickle.load(open(data_path + "df_coarse_phrase.pkl", "rb"))
    phrase_id = pickle.load(open(data_path + "phrase_id_coarse_map.pkl", "rb"))
    id_phrase_map = pickle.load(open(data_path + "id_phrase_coarse_map.pkl", "rb"))
    df = preprocess_df(df)
    tokenizer = fit_get_tokenizer(df.text, max_words=150000)
    # child_to_parent = pickle.load(open(data_path + "child_to_parent.pkl", "rb"))
    parent_to_child = pickle.load(open(data_path + "parent_to_child.pkl", "rb"))

    parent_labels = ["sports", "arts", "science"]
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    words = {}
    threshold = {}
    probability = {}
    for p in parent_labels:
        temp_df = df[df.label.isin([p])].reset_index(drop=True)
        tokenizer = fit_get_tokenizer(temp_df.text, max_words=150000)
        for ch in parent_to_child[p]:
            words[ch] = {}
            child_label_str = encode_phrase(" ".join([t for t in ch.split("_") if t not in stop_words]).strip(),
                                            phrase_id)
            if func == "cond_prob":
                thresh = get_conditional_probability(temp_df.text, child_label_str, encode_phrase(p, phrase_id))
                prob = get_conditional_probability_words(temp_df.text, child_label_str, tokenizer)
            elif func == "pmi":
                thresh = get_pmi(temp_df.text, child_label_str, encode_phrase(p, phrase_id))
                prob = get_pmi_words(temp_df.text, child_label_str, tokenizer)
                if thresh < 0:
                    thresh = 0
            print("Threshold for ", p, ch, str(thresh))
            for tok in tokenizer.word_index:
                assert tok in prob
                if prob[tok] >= thresh:
                    words[ch][decipher_phrase(tok, id_phrase_map)] = prob[tok]
            threshold[ch] = thresh
            probability[ch] = prob

    for p in parent_labels:
        temp_df = df[df.label.isin([p])].reset_index(drop=True)
        tokenizer = fit_get_tokenizer(temp_df.text, max_words=150000)
        parent_label_str = encode_phrase(" ".join([t for t in p.split("_") if t not in stop_words]).strip(), phrase_id)
        if func == "cond_prob":
            prob = get_conditional_probability_words(temp_df.text, parent_label_str, tokenizer)
        elif func == "pmi":
            prob = get_pmi_words(temp_df.text, parent_label_str, tokenizer)
        probability[p] = prob

    # words has deciphered phrase whereas probability has encoded phrase.
    for p in parent_labels:
        for ch in parent_to_child[p]:
            # siblings = set(words.keys()) - {ch}
            siblings = set(parent_to_child[p]) - {ch}
            uncles = set(parent_labels) - {p}
            cousins = set(words.keys()) - set(parent_to_child[p])
            removed_words = []
            for word in words[ch]:
                encoded_word = encode_phrase(word, phrase_id)
                flag = 0
                for sb in siblings:
                    try:
                        if probability[sb][encoded_word] >= words[ch][word] or probability[sb][encoded_word] >= \
                                threshold[sb]:
                            removed_words.append(word)
                            flag = 1
                            break
                    except:
                        continue

                if flag == 1:
                    continue

                for cousin in cousins:
                    try:
                        if probability[cousin][encoded_word] >= words[ch][word] or \
                                probability[cousin][encoded_word] >= threshold[cousin]:
                            removed_words.append(word)
                            flag = 1
                            break
                    except:
                        continue

                if flag == 1:
                    continue

                for sb in siblings:
                    try:
                        if probability[sb][encoded_word] >= probability[p][encoded_word]:
                            removed_words.append(word)
                            flag = 1
                            break
                    except:
                        continue

                if flag == 1:
                    continue

                for lbl in uncles:
                    try:
                        if probability[lbl][encoded_word] >= probability[p][encoded_word]:
                            removed_words.append(word)
                            flag = 1
                            break
                    except:
                        continue

                if flag == 1:
                    continue

                for lbl in uncles:
                    for children_uncle in parent_to_child[lbl]:
                        try:
                            if probability[children_uncle][encoded_word] >= probability[lbl][encoded_word]:
                                removed_words.append(word)
                                flag = 1
                                break
                        except:
                            continue

            for w in removed_words:
                words[ch].pop(w, None)

    for ch in words:
        words[ch] = {k: v for k, v in sorted(words[ch].items(), key=lambda item: -item[1])}
    if func == "cond_prob":
        json.dump(words, open(data_path + "conditional_prob_doc_all_filters.json", "w"))
    elif func == "pmi":
        json.dump(words, open(data_path + "pmi_doc_all_filters.json", "w"))
