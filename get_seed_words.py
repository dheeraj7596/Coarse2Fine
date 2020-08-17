import pickle
from classifier_seedwords import preprocess_df
from sklearn.feature_extraction.text import CountVectorizer
from parse_autophrase_output import decrypt
import json
from nltk.corpus import stopwords
from util import *


def get_rank_matrix(docfreq, inv_docfreq, label_docs_dict, doc_freq_thresh=5):
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    components = {}
    for l in label_docs_dict:
        components[l] = {}
        docs = label_docs_dict[l]
        docfreq_local = calculate_doc_freq(docs)
        vect = CountVectorizer(tokenizer=lambda x: x.split(), stop_words=stop_words, min_df=5)
        X = vect.fit_transform(docs)
        X_arr = X.toarray()
        rel_freq = np.sum(X_arr, axis=0) / len(docs)
        names = vect.get_feature_names()
        for i, name in enumerate(names):
            try:
                if docfreq_local[name] < doc_freq_thresh:
                    continue
            except:
                continue
            components[l][name] = {
                "reldocfreq": docfreq_local[name] / docfreq[name],
                "idf": inv_docfreq[name],
                "rel_freq": np.tanh(rel_freq[i]),
                "rank": (docfreq_local[name] / docfreq[name]) * inv_docfreq[name] * np.tanh(rel_freq[i])
            }
    return components


def diff_seeds(df, topk=100):
    labels = set(df.label)
    label_words = {}
    for i, row in df.iterrows():
        doc = row["text"]
        label = row["label"]
        words = doc.strip().split()
        for word in words:
            try:
                temp = label_words[label]
            except:
                label_words[label] = {}
            try:
                label_words[label][word] += 1
            except:
                label_words[label][word] = 1

    label_sort_words = {}
    for out_label in labels:
        label_sort_words[out_label] = {}
        for word in label_words[out_label]:
            count = 0
            for in_label in labels:
                if out_label == in_label:
                    continue
                try:
                    count += label_words[in_label][word]
                except:
                    continue
            label_sort_words[out_label][word] = label_words[out_label][word] - count

    topk_seeds = {}
    for l in labels:
        label_sort_words[l] = {k: v for k, v in sorted(label_sort_words[l].items(), key=lambda x: -x[1])}
        topk_seeds[l] = {k: v for k, v in list(label_sort_words[l].items())[:topk]}
    json.dump(topk_seeds, open(data_path + "top100_diff_words.json", "w"))
    json.dump(label_sort_words, open(data_path + "diff_seeds.json", "w"))


def decipher_phrase(word, id_phrase_map):
    id = decrypt(word)
    if id == None or id not in id_phrase_map:
        if word.startswith("fnust"):
            num_str = word[5:]
            flag = 0
            for index, char in enumerate(num_str):
                if index >= 5:
                    break
                try:
                    temp_int = int(char)
                    flag = 1
                except:
                    break
            if flag == 1:
                if int(num_str[:index]) in id_phrase_map:
                    print("I am here")
                    return id_phrase_map[int(num_str[:index])]
            else:
                raise ValueError("Something unexpected found: ", word)
        else:
            return word
    else:
        return id_phrase_map[id]
    return word


def conwea_seeds(df, id_phrase_map, topk=100):
    docfreq = calculate_df_doc_freq(df)
    inv_docfreq = calculate_inv_doc_freq(df, docfreq)
    label_docs_dict = get_label_docs_dict(df)
    components = get_rank_matrix(docfreq, inv_docfreq, label_docs_dict, doc_freq_thresh=5)
    all_seeds = {}
    topk_words = {}
    topk_phrases = {}

    for l in label_docs_dict:
        components[l] = {k: v for k, v in sorted(components[l].items(), key=lambda x: -x[1]["rank"])}

        topk_words[l] = {}
        for k, v in list(components[l].items()):
            if len(topk_words[l]) >= topk:
                break
            if decrypt(k) is None:
                topk_words[l][k] = v["rank"]

        topk_phrases[l] = {}
        for k, v in list(components[l].items()):
            if len(topk_phrases[l]) >= topk:
                break
            if decrypt(k) is not None:
                topk_phrases[l][decipher_phrase(k, id_phrase_map)] = v["rank"]

        all_seeds[l] = {k: v["rank"] for k, v in components[l].items()}
    json.dump(topk_words, open(data_path + "conwea_top100words.json", "w"))
    json.dump(topk_phrases, open(data_path + "conwea_top100phrases.json", "w"))
    json.dump(all_seeds, open(data_path + "conwea_seeds_words_phrases.json", "w"))


if __name__ == "__main__":
    base_path = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    dataset = "nyt"
    data_path = base_path + dataset + "/"

    df = pickle.load(open(data_path + "df_coarse_phrase.pkl", "rb"))
    id_phrase_map = pickle.load(open(data_path + "id_phrase_coarse_map.pkl", "rb"))
    df = preprocess_df(df)
    # diff_seeds(df)
    conwea_seeds(df, id_phrase_map)
