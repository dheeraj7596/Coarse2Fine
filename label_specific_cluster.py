import json
import pickle
import os
from util import cosine_similarity
from scipy.special import softmax
import numpy as np
from util import print_seed_dict
import sys
import string
from nltk.corpus import stopwords


def get_arg_max_label(ph, child_labels, embeddings, threshold=0.0):
    filtered_phrase = ph.translate(translator)
    sims = []
    for ch in child_labels:
        try:
            child_label_str = " ".join([t for t in ch.split("_") if t not in stop_words]).strip()
            sims.append(cosine_similarity(embeddings[filtered_phrase], embeddings[child_label_str]))
        except Exception as e:
            print("Error while computing cosine sim", e)
            return None
    sim_softmax = softmax(np.array(sims))
    if max(sim_softmax) >= threshold:
        max_ind = np.argmax(sim_softmax)
        return child_labels[max_ind]
    else:
        return None


def get_our_algo_arg_max(ph, child_labels, embeddings, mean_sim, threshold=0.0):
    filtered_phrase = ph.translate(translator)
    sims = []
    for ch in child_labels:
        try:
            child_label_str = " ".join([t for t in ch.split("_") if t not in stop_words]).strip()
            sims.append(cosine_similarity(embeddings[filtered_phrase], embeddings[child_label_str]) / mean_sim[ch])
        except Exception as e:
            print("Error while computing cosine sim", e)
            return None
    sim_softmax = softmax(np.array(sims))
    if max(sim_softmax) >= threshold:
        max_ind = np.argmax(sim_softmax)
        return child_labels[max_ind]
    else:
        return None


if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    # thresh = float(sys.argv[1])
    # algo = int(sys.argv[2])

    thresh = 0.2
    algo = 1

    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')

    label_embeddings = pickle.load(open(pkl_dump_dir + "label_bert_word_phrase_embeddings.pkl", "rb"))
    seed_phrases = json.load(open(pkl_dump_dir + "conwea_top100phrases.json", "r"))
    mean_sim = json.load(open(pkl_dump_dir + "mean_sim_label_specific.json", "r"))
    # mean_sim = json.load(open(pkl_dump_dir + "mean_sim_top100.json", "r"))
    all_sims = json.load(open(pkl_dump_dir + "all_sims_label_specific.json", "r"))

    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))

    child_seeds_dict = {}
    general_seeds = []
    all_child_labels = []
    for p in parent_to_child:
        embeddings = label_embeddings[p]
        child_labels = parent_to_child[p]
        all_child_labels += child_labels
        for ph in list(seed_phrases[p].keys()):
            if algo == 1:
                ch = get_arg_max_label(ph, child_labels, embeddings)
            elif algo == 2:
                ch = get_arg_max_label(ph, child_labels, embeddings, threshold=thresh)
            elif algo == 3:
                ch = get_our_algo_arg_max(ph, child_labels, embeddings, mean_sim)
            elif algo == 4:
                ch = get_our_algo_arg_max(ph, child_labels, embeddings, mean_sim, threshold=thresh)
            else:
                raise ValueError("algo should be in 1,2,3,4")

            if ch is None:
                general_seeds.append(ph)
            else:
                try:
                    child_seeds_dict[ch].append(ph)
                except:
                    child_seeds_dict[ch] = [ph]

    print_seed_dict(child_seeds_dict)
    print("Missing seeds for: ", set(all_child_labels) - set(child_seeds_dict.keys()))
    print("General Seeds: ", general_seeds)

    json.dump(child_seeds_dict, open(pkl_dump_dir + "child_seeds_dict_label_specific_argmax.json", "w"))
    # json.dump(child_seeds_dict, open(pkl_dump_dir + "child_seeds_dict_label_specific_argmax_thresh0.7.json", "w"))

    # json.dump(child_seeds_dict, open(pkl_dump_dir + "child_seeds_dict_label_specific_ouralgo_argmax.json", "w"))
    # json.dump(child_seeds_dict, open(pkl_dump_dir + "child_seeds_dict_label_specific_ouralgo_argmax_thresh0.2.json", "w"))

    # json.dump(child_seeds_dict, open(pkl_dump_dir + "child_seeds_dict_label_specific_ouralgo_argmax_meansim_100.json", "w"))
    # json.dump(child_seeds_dict, open(pkl_dump_dir + "child_seeds_dict_label_specific_ouralgo_argmax_meansim_100_thresh0.2.json", "w"))
