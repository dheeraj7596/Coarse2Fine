import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import string
import numpy as np
from util import *
import random
import torch
from transformers import BertModel, BertTokenizer
from sklearn.linear_model import LogisticRegression

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def get_bert_embedding(word):
    global bert_model, bert_tokenizer
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
    return sentence_embedding


def preprocess_df(df):
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    preprocessed_sentences = []
    preprocessed_labels = []
    for i, row in df.iterrows():
        label = row["label"]
        sent = row["text"]
        sent_nopuncts = sent.translate(translator)
        words_list = sent_nopuncts.strip().split()
        filtered_words = [word for word in words_list if word not in stop_words and len(word) != 1]
        preprocessed_sentences.append(" ".join(filtered_words))
        preprocessed_labels.append(label)
    df["text"] = preprocessed_sentences
    df["label"] = preprocessed_labels
    return df


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
                "similarity": cosine_similarity(get_bert_embedding(l).detach().cpu().numpy(),
                                                get_bert_embedding(name).detach().cpu().numpy())
            }
    return components


def compute_on_demand_feature(df, parent_label, child_label_str, sim=None):
    temp_df = df[df.label.isin([parent_label])]

    docfreq_all = 0
    docfreq_local = 0
    for i, row in df.iterrows():
        sent = row["text"]
        label = row["label"]
        if child_label_str in sent:
            docfreq_all += 1
            if label == parent_label:
                docfreq_local += 1

    count = 0
    for sent in temp_df.text:
        count += sent.count(child_label_str)

    reldocfreq = docfreq_local / docfreq_all
    idf = np.log(len(df) / docfreq_all)
    rel_freq = np.tanh(count / len(temp_df))
    if sim is None:
        return [reldocfreq, idf, rel_freq]
    else:
        similarity = cosine_similarity(get_bert_embedding(parent_label).detach().cpu().numpy(),
                                       get_bert_embedding(child_label_str).detach().cpu().numpy())

        return [reldocfreq, idf, rel_freq, similarity]


def get_feature(df, parent_label, child_label, components, sim=None):
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')

    if "_" in child_label:
        child_label_str = " ".join([t for t in child_label.split("_") if t not in stop_words]).strip()
        temp_features = compute_on_demand_feature(df, parent_label, child_label_str, sim)
    elif child_label in components[parent_label]:
        temp_dic = components[parent_label][child_label]
        if sim is None:
            temp_features = [temp_dic["reldocfreq"], temp_dic["idf"], temp_dic["rel_freq"]]
        else:
            temp_features = [temp_dic["reldocfreq"], temp_dic["idf"], temp_dic["rel_freq"], temp_dic["similarity"]]
    else:
        temp_features = compute_on_demand_feature(df, parent_label, child_label, sim)
    return temp_features


def generate_negative_features(parent_label, parent_to_child, components, df):
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    vect = CountVectorizer(tokenizer=lambda x: x.split(), stop_words=stop_words, min_df=5)

    pos_df = df[df.label.isin([parent_label])]
    X = vect.fit_transform(pos_df["text"])
    in_vocab = vect.get_feature_names()
    in_vocab = list(set(in_vocab) - set(parent_to_child[parent_label]))

    neg_df = df[~df.label.isin([parent_label])]
    X = vect.fit_transform(neg_df["text"])
    out_vocab = vect.get_feature_names()

    features = []
    labels = []

    for c in parent_to_child[parent_label]:
        # coin toss:
        # if p < 0.25 one negative example from in vocab,
        # if 0.25 < p < 0.5 one negative example from out vocab,
        # if 0.5 < p < 0.75 two negative examples from in vocab
        # if 0.75 < p < 1 two negative examples, one from in vocab and one from out vocab

        p = random.uniform(0, 1)

        if p < 0.25:
            neg_child_label = random.choice(in_vocab)
            temp_feature = get_feature(df, parent_label, neg_child_label, components)
            if len(temp_feature) > 0:
                features.append(temp_feature)
                labels.append(0)
        elif p < 0.5:
            neg_word = random.choice(out_vocab)
            temp_feature = get_feature(df, parent_label, neg_word, components)
            if len(temp_feature) > 0:
                features.append(temp_feature)
                labels.append(0)
        elif p < 0.75:
            neg_child_label = random.choice(in_vocab)
            temp_feature = get_feature(df, parent_label, neg_child_label, components)
            if len(temp_feature) > 0:
                features.append(temp_feature)
                labels.append(0)

            neg_child_label = random.choice(in_vocab)
            temp_feature = get_feature(df, parent_label, neg_child_label, components)
            if len(temp_feature) > 0:
                features.append(temp_feature)
                labels.append(0)
        else:
            neg_child_label = random.choice(in_vocab)
            temp_feature = get_feature(df, parent_label, neg_child_label, components)
            if len(temp_feature) > 0:
                features.append(temp_feature)
                labels.append(0)

            neg_word = random.choice(out_vocab)
            temp_feature = get_feature(df, parent_label, neg_word, components)
            if len(temp_feature) > 0:
                features.append(temp_feature)
                labels.append(0)
    return features, labels


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

    docfreq = calculate_df_doc_freq(df)
    inv_docfreq = calculate_inv_doc_freq(df, docfreq)
    label_docs_dict = get_label_docs_dict(df)

    # components = get_rank_matrix(docfreq, inv_docfreq, label_docs_dict, doc_freq_thresh=5)
    components = pickle.load(open(data_path + "components.pkl", "rb"))
    features = []
    seed_labels = []

    for parent_label in parent_to_child:
        for child_label in parent_to_child[parent_label]:
            temp_feature = get_feature(df, parent_label, child_label, components, sim=None)
            if len(temp_feature) > 0:
                features.append(temp_feature)
                seed_labels.append(1)

    for parent_label in parent_to_child:
        temp_features, temp_labels = generate_negative_features(parent_label, parent_to_child, components, df)
        features += temp_features
        seed_labels += temp_labels

    assert len(features) == len(seed_labels)
    print("Number of training samples: ", len(features))
    feature_vec = np.array(features)
    label_vec = np.array(seed_labels)
    print(feature_vec)
    print(label_vec)
    clf = LogisticRegression().fit(feature_vec, label_vec)
    preds = clf.predict(feature_vec)
    print("Training Performance")
    print(classification_report(seed_labels, preds))
    pickle.dump(clf, open(data_path + "model_dumps/clf_logreg_nosim.pkl", "wb"))
