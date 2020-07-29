import pickle
from nltk.corpus import stopwords
import numpy as np
from util import *
from transformers import BertModel, BertTokenizer
from classifier_seedwords import preprocess_df
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def get_embedding(word):
    global bert_model, bert_tokenizer, embeddings
    try:
        return embeddings[word]
    except:
        word_ids = bert_tokenizer.encode(word)
        word_ids = torch.LongTensor(word_ids)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        bert_model = bert_model.to(device)
        word_ids = word_ids.to(device)
        bert_model.eval()
        word_ids = word_ids.unsqueeze(0)
        out = bert_model(input_ids=word_ids)
        hidden_states = out[2]
        sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
        embeddings[word] = sentence_embedding
    return sentence_embedding


def compute_on_demand_fine_feature(df, parent_label, child_label_str, candidate_word):
    docfreq_all = 0  # number of documents where candidate_word appears in df
    docfreq_coarse = 0  # number of documents with parent_label as label and candidate_word appears in the documents
    docfreq_local = 0  # number of documents with parent_label as label and child_label_str appears in the documents and also candidate_word appears in the documents

    count = 0
    total_len = 0
    for i, row in df.iterrows():
        sent = row["text"]
        label = row["label"]
        if candidate_word in sent:
            docfreq_all += 1

        if label == parent_label:
            if child_label_str in sent:
                total_len += 1
            if candidate_word in sent:
                docfreq_coarse += 1
            if child_label_str in sent and candidate_word in sent:
                docfreq_local += 1
                count += sent.count(candidate_word)

    reldocfreq = docfreq_local / docfreq_coarse
    idf = np.log(len(df) / docfreq_all)
    rel_freq = np.tanh(count / total_len)
    similarity = cosine_similarity(get_embedding(candidate_word).detach().cpu().numpy(),
                                   get_embedding(child_label_str).detach().cpu().numpy())

    return [reldocfreq, idf, rel_freq, similarity]


def get_seeds(df, parent_label, child_label, clf, topk=20):
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    child_label_str = " ".join([t for t in child_label.split("_") if t not in stop_words]).strip()

    pos_df = df[df.label.isin([parent_label])]
    candidate_words = set()

    for sent in pos_df.text:
        if child_label_str in sent:
            candidate_words.update(set(sent.split()))

    candidate_words = candidate_words - {child_label_str}

    candidate_words = list(candidate_words)
    features = []
    for word in candidate_words:
        temp_features = compute_on_demand_fine_feature(df, parent_label, child_label_str, word)
        features.append(temp_features)

    feature_vec = np.array(features)
    probs = clf.predict_proba(feature_vec)
    pos_probs = probs[:, 1]
    inds = pos_probs.argsort()[::-1][:topk]
    seeds = []
    for i in inds:
        seeds.append((candidate_words[i], probs[i, 1]))
    return seeds


if __name__ == "__main__":
    base_path = "/data4/dheeraj/coarse2fine/"
    # base_path = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    dataset = "nyt"
    data_path = base_path + dataset + "/"

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

    df = pickle.load(open(data_path + "df_coarse.pkl", "rb"))
    child_to_parent = pickle.load(open(data_path + "child_to_parent.pkl", "rb"))
    parent_to_child = pickle.load(open(data_path + "parent_to_child.pkl", "rb"))
    df = preprocess_df(df)

    clf = pickle.load(open(data_path + "clf_logreg.pkl", "rb"))
    embeddings = {}

    fine_seeds = {}
    for p in parent_to_child:
        print("Parent Label: ", p)
        print("*" * 40)
        for c in parent_to_child[p]:
            print("Child Label: ", c)
            print("#" * 40)
            fine_seeds[c] = get_seeds(df, p, c, clf, topk=20)
            for s in fine_seeds[c]:
                print(s)
            print("*" * 80)
