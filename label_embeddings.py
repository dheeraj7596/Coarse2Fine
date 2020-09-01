import pickle
from transformers import BertTokenizer, BertModel
import pickle
import os
import torch
from classifier_seedwords import preprocess_df
import numpy as np
from nltk.corpus import stopwords
import string
from add_phrase_embeddings import get_phrase_bert_embeddings

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    df = pickle.load(open(pkl_dump_dir + "df_coarse.pkl", "rb"))
    df = preprocess_df(df)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))

    label_embeddings = {}
    count = {}
    for p in parent_to_child:
        temp_df = df[df.label.isin([p])]
        temp_df = temp_df.reset_index(drop=True)

        for ch in parent_to_child[p]:
            child_label_str = " ".join([t for t in ch.split("_") if t not in stop_words]).strip()
            label_embeddings, count = get_phrase_bert_embeddings(label_embeddings, count, model, tokenizer,
                                                                 [child_label_str], list(temp_df.text))

    for w in label_embeddings:
        try:
            label_embeddings[w] = label_embeddings[w] / count[w]
        except:
            continue

    pickle.dump(label_embeddings, open(pkl_dump_dir + "label_embeddings.pkl", "wb"))
