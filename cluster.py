import json
from transformers import BertForSequenceClassification, BertTokenizer, BertModel
import pickle
import os
import torch
from util import cosine_similarity
from scipy.special import softmax
import numpy as np
from util import print_seed_dict
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def get_bert_embeddings(model, tokenizer, words):
    embeddings = {}
    batch_word_ids = tokenizer(words, padding=True, truncation=True, return_tensors="pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_word_ids = batch_word_ids.to(device)
    out = model(**batch_word_ids)
    # hidden_states = out[1]
    hidden_states = out[2]
    sentence_embeddings = torch.mean(hidden_states[-1], dim=1)
    for i, word in enumerate(words):
        embeddings[word] = sentence_embeddings[i, :].detach().cpu().numpy()
    return embeddings


def get_fine_seeds(child_labels, candidate_phrases, model, tokenizer, thresh=0.7):
    child_label_embeddings = get_bert_embeddings(model, tokenizer, child_labels)
    candidate_phrases_embeddings = get_bert_embeddings(model, tokenizer, candidate_phrases)

    fine_seeds_dict = {}
    for ch in child_labels:
        fine_seeds_dict[ch] = []

    for cand_word in candidate_phrases_embeddings:
        temp_sim = []
        for ch in child_labels:
            temp_sim.append(cosine_similarity(child_label_embeddings[ch], candidate_phrases_embeddings[cand_word]))
        sim_softmax = softmax(np.array(temp_sim))
        if max(sim_softmax) >= thresh:
            max_ind = np.argmax(sim_softmax)
            fine_seeds_dict[child_labels[max_ind]].append(cand_word)
    return fine_seeds_dict


if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    tok_path = pkl_dump_dir + "bert/tokenizer"
    model_path = pkl_dump_dir + "bert/model"

    thresh = float(sys.argv[1])
    seed_phrases = json.load(open(pkl_dump_dir + "conwea_top100phrases.json", "r"))

    # tokenizer = BertTokenizer.from_pretrained(tok_path)
    # model = BertForSequenceClassification.from_pretrained(model_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))

    child_seeds_dict = {}
    for p in parent_to_child:
        child_labels = parent_to_child[p]
        fine_seeds_dict = get_fine_seeds(child_labels, list(seed_phrases[p].keys()), model, tokenizer, thresh=thresh)
        for c in child_labels:
            child_seeds_dict[c] = fine_seeds_dict[c]

    print_seed_dict(child_seeds_dict)
