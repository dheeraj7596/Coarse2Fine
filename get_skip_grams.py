import json
from classifier_seedwords import preprocess_df
from sklearn.mixture import GaussianMixture
from gensim.models import word2vec
from get_seed_words import decipher_phrase
import numpy as np
import pickle


def encode_phrase(entity, phrase_id):
    try:
        id = phrase_id[entity]
        return "fnust" + str(id)
    except:
        return entity


def encode_phrases(entities, phrase_id):
    for i, s in enumerate(entities):
        entities[i] = encode_phrase(s, phrase_id)
    return entities


def get_all_skipgrams(df):
    def add_to_dic(word_skip_gram_dict, skip_gram_word_dict, tokens, word_ind, skipgram_inds):
        skipgram = []
        for i in skipgram_inds:
            if i == "placeholder":
                skipgram.append(i)
                continue
            if i < 0 or i >= len(tokens):
                return word_skip_gram_dict, skip_gram_word_dict
            else:
                skipgram.append(tokens[i])

        skipgram = " ".join(skipgram)
        word = tokens[word_ind]

        try:
            word_skip_gram_dict[word].add(skipgram)
        except:
            word_skip_gram_dict[word] = {skipgram}
        try:
            skip_gram_word_dict[skipgram].add(word)
        except:
            skip_gram_word_dict[skipgram] = {word}
        return word_skip_gram_dict, skip_gram_word_dict

    place_holder = "placeholder"
    label_word_skip_gram_dict = {}
    label_skip_gram_word_dict = {}
    labels = list(set(df.label))

    for l in labels:
        word_skip_gram_dict = {}  # map from word to its respective skip-grams
        skip_gram_word_dict = {}  # map from skip-gram to its respective words
        temp_df = df[df.label.isin([l])]
        temp_df = temp_df.reset_index(drop=True)
        for i, row in temp_df.iterrows():
            sent = row["text"]
            tokens = sent.strip().split()
            for j in range(len(tokens)):
                word_skip_gram_dict, skip_gram_word_dict = add_to_dic(word_skip_gram_dict, skip_gram_word_dict, tokens,
                                                                      j, [j - 1, place_holder, j + 1])
                word_skip_gram_dict, skip_gram_word_dict = add_to_dic(word_skip_gram_dict, skip_gram_word_dict, tokens,
                                                                      j, [j - 2, j - 1, place_holder, j + 1])
                word_skip_gram_dict, skip_gram_word_dict = add_to_dic(word_skip_gram_dict, skip_gram_word_dict, tokens,
                                                                      j, [j - 3, j - 2, j - 1, place_holder, j + 1])
                word_skip_gram_dict, skip_gram_word_dict = add_to_dic(word_skip_gram_dict, skip_gram_word_dict, tokens,
                                                                      j, [j - 1, place_holder, j + 1, j + 2, j + 3])
                word_skip_gram_dict, skip_gram_word_dict = add_to_dic(word_skip_gram_dict, skip_gram_word_dict, tokens,
                                                                      j, [j - 2, j - 1, place_holder, j + 1, j + 2])
                word_skip_gram_dict, skip_gram_word_dict = add_to_dic(word_skip_gram_dict, skip_gram_word_dict, tokens,
                                                                      j, [j - 1, place_holder, j + 1, j + 2])
        label_word_skip_gram_dict[l] = word_skip_gram_dict
        label_skip_gram_word_dict[l] = skip_gram_word_dict
    return label_word_skip_gram_dict, label_skip_gram_word_dict


def get_skipgrams(encoded_entities, word_skip_gram_dict, skip_gram_word_dict, min_thresh=5, max_thresh=50):
    entity_skipgrams_dic = {}
    for e in encoded_entities:
        for sg in word_skip_gram_dict[e]:
            if len(skip_gram_word_dict[sg]) >= min_thresh and len(skip_gram_word_dict[sg]) <= max_thresh:
                try:
                    entity_skipgrams_dic[e].add(sg)
                except:
                    entity_skipgrams_dic[e] = {sg}
    return entity_skipgrams_dic


def embed_skipgrams(label_skipgrams, embedding_model):
    embedded_label_skipgrams = []
    for sg in label_skipgrams:
        temp = sg.strip().split()
        vec = np.zeros(embedding_model["apple"].shape)
        for w in temp:
            if w == "placeholder":
                continue
            vec += embedding_model[w]
        vec = vec / (len(temp) - 1)
        embedded_label_skipgrams.append(vec)
    return embedded_label_skipgrams


def update_label_skipgram_clusters(label_skipgram_clusters, label_skipgrams, idx, skip_gram_word_dict,
                                   skipgram_entities, id_phrase_map, label, type="all"):
    for i, sg in enumerate(label_skipgrams):
        sg_decoded = []
        for w in sg.strip().split():
            if w == "placeholder":
                sg_decoded.append(w)
            else:
                w_decoded = decipher_phrase(w, id_phrase_map)
                sg_decoded.append(w_decoded)
        sg_decoded = " ".join(sg_decoded)

        cluster_id = int(idx[i])
        try:
            r = label_skipgram_clusters[label][cluster_id]
        except:
            label_skipgram_clusters[label][cluster_id] = {}

        if type == "all":
            # This gets the results from all vocab
            for w in skip_gram_word_dict[sg]:
                w_decoded = decipher_phrase(w, id_phrase_map)
                try:
                    label_skipgram_clusters[label][cluster_id][sg_decoded].append(w_decoded)
                except:
                    label_skipgram_clusters[label][cluster_id][sg_decoded] = [w_decoded]
        else:
            # This gets the results from only chosen seed words.
            for w in skipgram_entities[sg]:
                w_decoded = decipher_phrase(w, id_phrase_map)
                try:
                    label_skipgram_clusters[label][cluster_id][sg_decoded].append(w_decoded)
                except:
                    label_skipgram_clusters[label][cluster_id][sg_decoded] = [w_decoded]

    return label_skipgram_clusters


if __name__ == "__main__":
    base_path = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # base_path = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt"
    data_path = base_path + dataset + "/"

    df = pickle.load(open(data_path + "df_coarse_phrase.pkl", "rb"))
    df = preprocess_df(df)
    print("Loaded Dataframe..", flush=True)
    # tokenizer = fit_get_tokenizer(df.text, max_words=150000)
    label_words_phrases = json.load(open(data_path + "conwea_top100words_phrases.json", "r"))
    phrase_id = pickle.load(open(data_path + "phrase_id_coarse_map.pkl", "rb"))
    id_phrase_map = pickle.load(open(data_path + "id_phrase_coarse_map.pkl", "rb"))

    # label_word_skip_gram_dict, label_skip_gram_word_dict = get_all_skipgrams(df)
    # pickle.dump(label_word_skip_gram_dict, open(data_path + "label_word_skip_gram_dict.pkl", "wb"))
    # pickle.dump(label_skip_gram_word_dict, open(data_path + "label_skip_gram_word_dict.pkl", "wb"))

    label_word_skip_gram_dict = pickle.load(open(data_path + "label_word_skip_gram_dict.pkl", "rb"))
    label_skip_gram_word_dict = pickle.load(open(data_path + "label_skip_gram_word_dict.pkl", "rb"))

    print("Loaded skipgram maps..", flush=True)
    embedding_model = word2vec.Word2Vec.load(data_path + "word2vec.model")
    print("Loaded word2vec model..", flush=True)
    label_skipgram_clusters = {}
    for label in label_words_phrases:
        word_skip_gram_dict = label_word_skip_gram_dict[label]
        skip_gram_word_dict = label_skip_gram_word_dict[label]
        print("Getting skipgrams for ", label, flush=True)
        label_skipgram_clusters[label] = {}
        clf = GaussianMixture(n_components=10, covariance_type="tied", init_params='kmeans', max_iter=50)
        entities = list(label_words_phrases[label].keys())
        encoded_entities = encode_phrases(entities, phrase_id)
        entity_skipgrams = get_skipgrams(encoded_entities, word_skip_gram_dict, skip_gram_word_dict, min_thresh=5,
                                         max_thresh=50)
        skipgram_entities = {}
        for e in entity_skipgrams:
            for sg in entity_skipgrams[e]:
                try:
                    skipgram_entities[sg].append(e)
                except:
                    skipgram_entities[sg] = [e]

        label_skipgrams = list(skipgram_entities.keys())
        embedded_label_skipgrams = embed_skipgrams(label_skipgrams, embedding_model)

        print("Clustering skipgrams..", flush=True)
        clf.fit(embedded_label_skipgrams)
        idx = clf.predict(embedded_label_skipgrams)
        label_skipgram_clusters = update_label_skipgram_clusters(label_skipgram_clusters, label_skipgrams, idx,
                                                                 skip_gram_word_dict, skipgram_entities, id_phrase_map,
                                                                 label)

    pickle.dump(label_skipgram_clusters, open(data_path + "label_skipgram_clusters.pkl", "wb"))
    json.dump(label_skipgram_clusters, open(data_path + "label_skipgram_clusters.json", "w"))
