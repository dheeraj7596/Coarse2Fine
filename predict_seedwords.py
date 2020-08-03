import pickle
from nltk.corpus import stopwords
from util import *
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score
from classifier_seedwords import preprocess_df
import torch
import pandas as pd
import os
import json
import nltk

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


def compute_on_demand_fine_feature(df, parent_label, child_label_str, candidate_word, sim=None):
    candidate_word_in_df = df[df['text'].str.contains(candidate_word)]
    parent_label_df = df[df["label"] == parent_label]
    child_label_str_in_parentdf = parent_label_df[parent_label_df['text'].str.contains(child_label_str)]
    candidate_word_in_parentdf = parent_label_df[parent_label_df['text'].str.contains(candidate_word)]

    docfreq_all = len(candidate_word_in_df)  # number of documents where candidate_word appears in df
    total_len = len(child_label_str_in_parentdf)
    docfreq_coarse = len(
        candidate_word_in_parentdf)  # number of documents with parent_label as label and candidate_word appears in the documents

    child_label_candidate_word_in_parentdf = pd.merge(child_label_str_in_parentdf, candidate_word_in_parentdf,
                                                      how='inner', on=['text'])
    docfreq_local = len(
        child_label_candidate_word_in_parentdf)  # number of documents with parent_label as label and child_label_str appears in the documents and also candidate_word appears in the documents

    count = 0
    for i, row in child_label_candidate_word_in_parentdf.iterrows():
        sent = row["text"]
        count += sent.count(candidate_word)

    reldocfreq = docfreq_local / docfreq_coarse
    idf = np.log(len(df) / docfreq_all)
    rel_freq = np.tanh(count / total_len)
    if sim is None:
        return [reldocfreq, idf, rel_freq]
    else:
        similarity = cosine_similarity(get_embedding(candidate_word).detach().cpu().numpy(),
                                       get_embedding(child_label_str).detach().cpu().numpy())
        return [reldocfreq, idf, rel_freq, similarity]


def get_seeds(df, parent_label, child_label, clf, sim=None):
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    child_label_str = " ".join([t for t in child_label.split("_") if t not in stop_words]).strip()

    pos_df = df[df.label.isin([parent_label])]
    candidate_words = set()

    for sent in pos_df.text:
        if child_label_str in sent:
            words = sent.strip().split()
            is_noun = lambda pos: pos[:2] == 'NN'
            nouns = set([word for (word, pos) in nltk.pos_tag(words) if is_noun(pos)])
            candidate_words.update(nouns)

    candidate_words = candidate_words - {child_label_str}

    candidate_words = list(candidate_words)
    print("Number of candidate words: ", len(candidate_words), "for child label: ", child_label, "for parent label: ",
          parent_label)
    features = []
    for word in candidate_words:
        temp_features = compute_on_demand_fine_feature(df, parent_label, child_label_str, word, sim)
        features.append(temp_features)

    feature_vec = np.array(features)
    probs = clf.predict_proba(feature_vec)
    pos_probs = probs[:, 1]
    inds = pos_probs.argsort()[::-1]
    seeds = []
    for i in inds:
        seeds.append((candidate_words[i], probs[i, 1]))
    return seeds


def performance(pred_fine_seeds, actual_seeds, clf, parent_to_child, sim):
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')

    def precision(actual, predicted, k):
        act_set = set(actual)
        pred_set = set(predicted[:k])
        if k <= len(predicted):
            result = len(act_set & pred_set) / float(k)
        else:
            result = len(act_set & pred_set) / float(len(predicted))
        return result

    mean_prec_k = {3: 0, 10: 0, 20: 0}
    n = 0
    for parent in parent_to_child:
        for child in parent_to_child[parent]:
            mean_prec_k[3] = mean_prec_k[3] + precision(actual_seeds[child], pred_fine_seeds[child], 3)
            mean_prec_k[10] = mean_prec_k[10] + precision(actual_seeds[child], pred_fine_seeds[child], 10)
            mean_prec_k[20] = mean_prec_k[20] + precision(actual_seeds[child], pred_fine_seeds[child], 20)
            n += 1

    mean_prec_k[3] = mean_prec_k[3] / n
    mean_prec_k[10] = mean_prec_k[10] / n
    mean_prec_k[20] = mean_prec_k[20] / n

    print("Precision at 3: ", mean_prec_k[3])
    print("Precision at 10: ", mean_prec_k[10])
    print("Precision at 20: ", mean_prec_k[20])

    features = []
    for parent in parent_to_child:
        for child in parent_to_child[parent]:
            for word in actual_seeds[child]:
                child_label_str = " ".join([t for t in child.split("_") if t not in stop_words]).strip()
                features.append(compute_on_demand_fine_feature(df, parent, child_label_str, word, sim))

    feature_vec = np.array(features)
    preds = clf.predict(feature_vec)
    true = np.ones(preds.shape)
    print("Accuracy on Actual seeds: ", accuracy_score(true, preds))


if __name__ == "__main__":
    base_path = "/data4/dheeraj/coarse2fine/"
    # base_path = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    dataset = "nyt"
    data_path = base_path + dataset + "/"
    topk = 20
    sim = None

    # bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

    df = pickle.load(open(data_path + "df_coarse.pkl", "rb"))
    child_to_parent = pickle.load(open(data_path + "child_to_parent.pkl", "rb"))
    parent_to_child = pickle.load(open(data_path + "parent_to_child.pkl", "rb"))
    df = preprocess_df(df)

    actual_seeds = json.load(open(data_path + "actual_seedwords.json", "rb"))

    clf = pickle.load(open(data_path + "model_dumps/clf_logreg_nosim.pkl", "rb"))
    embeddings = {}

    fine_seeds = {}
    for p in parent_to_child:
        print("Parent Label: ", p)
        print("*" * 40)
        for c in parent_to_child[p]:
            print("Child Label: ", c)
            print("#" * 40)
            fine_seeds[c] = get_seeds(df, p, c, clf, sim)
            for s in fine_seeds[c][:topk]:
                print(s)
            print("*" * 80)

    performance(fine_seeds, actual_seeds, clf, parent_to_child, sim)
