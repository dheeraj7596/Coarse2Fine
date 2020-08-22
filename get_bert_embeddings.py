import json
from transformers import BertTokenizer, BertModel
import pickle
import os
import torch
from classifier_seedwords import preprocess_df
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def get_bert_embeddings(model, tokenizer, sentences):
    # concatenate last 4 layers for the word embedding

    embeddings = {}
    count = {}
    batch_word_ids = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_word_ids = batch_word_ids.to(device)
    out = model(**batch_word_ids)
    hidden_states = out[2]  # tuple of size num_attn heads = 13, each containing batch_size x num_tokens x dim
    batch_embeddings = torch.stack(hidden_states, dim=0)  # (13 x batch_size x num_tokens x dim)
    batch_embeddings = batch_embeddings.permute(1, 2, 0, 3)  # (batch_size x num_tokens x 13 x dim)
    for i, sent in enumerate(sentences):
        if i % 100 == 0:
            print("Finished sentences: ", i)
        sentence_embedding = batch_embeddings[i, :, :, :]  # (num_tokens x 13 x dim)
        total_tokens = sentence_embedding.shape[0]
        words = sent.strip().split()
        ind = 0
        for j, word in enumerate(words):
            num_tokens = len(tokenizer.tokenize(word))
            if ind >= total_tokens or ind + num_tokens > total_tokens:
                break
            word_embeddings = sentence_embedding[ind: ind + num_tokens, -4:, :]  # (num_tokens x 4 x dim)
            word_embeddings = word_embeddings.contiguous().view(num_tokens, -1)
            word_embeddings = torch.mean(word_embeddings, dim=0)
            ind += num_tokens
            try:
                embeddings[word] += word_embeddings.contiguous().detach().cpu().numpy()
                count[word] += 1
            except:
                embeddings[word] = word_embeddings.contiguous().detach().cpu().numpy()
                count[word] = 1

    for word in embeddings:
        embeddings[word] = embeddings[word] / count[word]
    return embeddings


if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_coarse.pkl", "rb"))
    df = preprocess_df(df)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))

    embeddings = get_bert_embeddings(model, tokenizer, list(df.text)[:2])
    pickle.dump(embeddings, open(pkl_dump_dir + "bert_word_embeddings.pkl", "wb"))
