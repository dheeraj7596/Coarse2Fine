import json
import pickle
import os
from util import cosine_similarity
from scipy.special import softmax
import numpy as np
from util import print_seed_dict, fit_get_tokenizer
from sklearn.metrics import classification_report
import sys
import string
from nltk.corpus import stopwords


def get_arg_max_label(ph, child_labels, embeddings, label_embeddings, threshold=0.0):
    filtered_phrase = ph.translate(translator)
    sims = []
    for ch in child_labels:
        try:
            child_label_str = " ".join([t for t in ch.split("_") if t not in stop_words]).strip()
            sims.append(cosine_similarity(embeddings[filtered_phrase], label_embeddings[child_label_str]))
        except Exception as e:
            print("Error while computing cosine sim", e)
            return None, None
    sim_softmax = softmax(np.array(sims))
    if max(sim_softmax) >= threshold:
        max_ind = np.argmax(sim_softmax)
        return child_labels[max_ind], max(sim_softmax)
    else:
        return None, None


def get_our_algo_arg_max(ph, child_labels, embeddings, label_embeddings, mean_sim, threshold=0.0):
    filtered_phrase = ph.translate(translator)
    sims = []
    for ch in child_labels:
        try:
            child_label_str = " ".join([t for t in ch.split("_") if t not in stop_words]).strip()
            sims.append(
                cosine_similarity(embeddings[filtered_phrase], label_embeddings[child_label_str]) / mean_sim[ch])
        except Exception as e:
            print("Error while computing cosine sim", e)
            return None, None
    sim_softmax = softmax(np.array(sims))
    if max(sim_softmax) >= threshold:
        max_ind = np.argmax(sim_softmax)
        return child_labels[max_ind], max(sim_softmax)
    else:
        return None, None


def get_our_algo_std_dev_arg_max(ph, child_labels, embeddings, label_embeddings, mean_sim, std_dev, threshold=0.0):
    filtered_phrase = ph.translate(translator)
    sims = []
    for ch in child_labels:
        try:
            child_label_str = " ".join([t for t in ch.split("_") if t not in stop_words]).strip()
            sims.append(
                (cosine_similarity(embeddings[filtered_phrase], label_embeddings[child_label_str]) - mean_sim[ch]) /
                std_dev[ch])
        except Exception as e:
            print("Error while computing cosine sim", e)
            return None, None
    sim_softmax = softmax(np.array(sims))
    if max(sim_softmax) >= threshold:
        max_ind = np.argmax(sim_softmax)
        return child_labels[max_ind], max(sim_softmax)
    else:
        return None, None


def generate_pseudo_labels(df, labels, label_term_dict, tokenizer):
    def argmax_label(count_dict):
        maxi = 0
        max_label = None
        for l in count_dict:
            count = 0
            for t in count_dict[l]:
                count += count_dict[l][t]
            if count > maxi:
                maxi = count
                max_label = l
        return max_label

    y = []
    X = []
    y_true = []
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w
    for index, row in df.iterrows():
        line = row["text"]
        label = row["label"]
        tokens = tokenizer.texts_to_sequences([line])[0]
        words = []
        for tok in tokens:
            words.append(index_word[tok])
        count_dict = {}
        flag = 0
        for l in labels:
            seed_words = set()
            for w in label_term_dict[l]:
                seed_words.add(w)
            int_labels = list(set(words).intersection(seed_words))
            if len(int_labels) == 0:
                continue
            for word in words:
                if word in int_labels:
                    flag = 1
                    try:
                        temp = count_dict[l]
                    except:
                        count_dict[l] = {}
                    try:
                        count_dict[l][word] += 1
                    except:
                        count_dict[l][word] = 1
        if flag:
            lbl = argmax_label(count_dict)
            if not lbl:
                continue
            y.append(lbl)
            X.append(line)
            y_true.append(label)
    return X, y, y_true


def top_k_seeds(child_seeds_dict, phrase_id, topk=3):
    label_term_dict = {}
    for ch in child_seeds_dict:
        label_term_dict[ch] = []
        child_seeds_dict[ch] = {k: v for (k, v) in sorted(child_seeds_dict[ch].items(), key=lambda y: -y[1])[:topk]}
        seeds = list(child_seeds_dict[ch].keys())
        for s in seeds:
            try:
                id = phrase_id[s]
                label_term_dict[ch].append("fnust" + str(id))
            except:
                label_term_dict[ch].append(s)
    return label_term_dict


def modify_seeds(child_seeds_dict, phrase_id):
    label_term_dict = {}
    for ch in child_seeds_dict:
        label_term_dict[ch] = []
        seeds = list(child_seeds_dict[ch].keys())
        for s in seeds:
            try:
                id = phrase_id[s]
                label_term_dict[ch].append("fnust" + str(id))
            except:
                label_term_dict[ch].append(s)
    return label_term_dict


if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    # thresh = float(sys.argv[1])
    # algo = int(sys.argv[2])

    thresh = 0.2
    algo = 5

    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')

    embeddings = pickle.load(open(pkl_dump_dir + "bert_word_phrase_embeddings.pkl", "rb"))
    label_embeddings = pickle.load(open(pkl_dump_dir + "label_embeddings.pkl", "rb"))
    seed_phrases = json.load(open(pkl_dump_dir + "conwea_top100phrases.json", "r"))
    # mean_sim = json.load(open(pkl_dump_dir + "mean_sim.json", "r"))
    std_dev = json.load(open(pkl_dump_dir + "std_dev_sim_label_top_words_labels.json", "r"))
    mean_sim = json.load(open(pkl_dump_dir + "mean_sim_label_top_words_labels.json", "r"))
    all_sims = json.load(open(pkl_dump_dir + "all_sims_label_top_words_labels.json", "r"))

    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    phrase_id = pickle.load(open(pkl_dump_dir + "phrase_id_coarse_map.pkl", "rb"))

    child_seeds_dict = {}
    general_seeds = []
    all_child_labels = []
    for p in parent_to_child:
        child_labels = parent_to_child[p]
        all_child_labels += child_labels
        for ph in list(seed_phrases[p].keys()):
            if algo == 1:
                ch, maxi = get_arg_max_label(ph, child_labels, embeddings, label_embeddings)
            elif algo == 2:
                ch, maxi = get_arg_max_label(ph, child_labels, embeddings, label_embeddings, threshold=thresh)
            elif algo == 3:
                ch, maxi = get_our_algo_arg_max(ph, child_labels, embeddings, label_embeddings, mean_sim)
            elif algo == 4:
                ch, maxi = get_our_algo_arg_max(ph, child_labels, embeddings, label_embeddings, mean_sim,
                                                threshold=thresh)
            elif algo == 5:
                ch, maxi = get_our_algo_std_dev_arg_max(ph, child_labels, embeddings, label_embeddings, mean_sim,
                                                        std_dev, threshold=thresh)
            else:
                raise ValueError("algo should be in 1,2,3,4")

            if ch is None:
                general_seeds.append(ph)
            else:
                try:
                    child_seeds_dict[ch][ph] = maxi
                except:
                    child_seeds_dict[ch] = {ph: maxi}

    print_seed_dict(child_seeds_dict)
    print("Missing seeds for: ", set(all_child_labels) - set(child_seeds_dict.keys()))
    print("General Seeds: ", general_seeds)

    # json.dump(child_seeds_dict, open(pkl_dump_dir + "child_seeds_dict_label.json", "w"))

    df_fine_phrase = pickle.load(open(pkl_dump_dir + "df_fine_phrase.pkl", "rb"))
    tokenizer = fit_get_tokenizer(df_fine_phrase.text, max_words=150000)
    label_term_dict = top_k_seeds(child_seeds_dict, phrase_id)
    # label_term_dict = modify_seeds(child_seeds_dict, phrase_id)
    X, y, y_true = generate_pseudo_labels(df_fine_phrase, list(set(df_fine_phrase.label)), label_term_dict, tokenizer)

    print(classification_report(y_true, y))
    pass

    # json.dump(child_seeds_dict, open(pkl_dump_dir + "child_seeds_dict_argmax_mean.json", "w"))
    # json.dump(child_seeds_dict, open(pkl_dump_dir + "child_seeds_dict_argmax_mean_thresh0.7.json", "w"))

    # json.dump(child_seeds_dict, open(pkl_dump_dir + "child_seeds_dict_ouralgo_argmax_new.json", "w"))
    # json.dump(child_seeds_dict, open(pkl_dump_dir + "child_seeds_dict_ouralgo_argmax_parent.json", "w"))
    # json.dump(child_seeds_dict, open(pkl_dump_dir + "child_seeds_dict_ouralgo_argmax_thresh0.2.json", "w"))

    # json.dump(child_seeds_dict, open(pkl_dump_dir + "child_seeds_dict_ouralgo_argmax_meansim_100.json", "w"))
    # json.dump(child_seeds_dict, open(pkl_dump_dir + "child_seeds_dict_ouralgo_argmax_meansim_100_thresh0.2.json", "w"))
