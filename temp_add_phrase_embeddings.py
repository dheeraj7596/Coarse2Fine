import pickle
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
import string
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def get_phrase_embedding(ph, embeddings, translator):
    temp = None
    mod_ph = ph.translate(translator)
    words = mod_ph.strip().split()
    den = len(words)
    for word in words:
        if len(word) == 1:
            den = den - 1
            continue
        try:
            var = embeddings[word]
        except:
            return None

        if temp is None:
            temp = embeddings[word]
        else:
            temp += embeddings[word]

    if temp is None:
        return temp

    temp = temp / den
    return temp


def bert_embedding(model, tokenizer, phrase):
    # concatenate last 4 layers for the word embedding
    batch_word_ids = tokenizer(phrase, padding=True, truncation=True, return_tensors="pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_word_ids = batch_word_ids.to(device)
    out = model(**batch_word_ids)
    hidden_states = out[2]  # tuple of size num_attn heads = 13, each containing batch_size x num_tokens x dim
    batch_embeddings = torch.stack(hidden_states, dim=0)  # (13 x batch_size x num_tokens x dim)
    batch_embeddings = batch_embeddings.permute(1, 2, 0, 3)  # (batch_size x num_tokens x 13 x dim)
    sentence_embedding = batch_embeddings.squeeze(0)
    word_embeddings = sentence_embedding[:, -4:, :]  # (num_tokens x 4 x dim)
    word_embeddings = word_embeddings.contiguous().view(word_embeddings.shape[0], -1)
    word_embeddings = torch.mean(word_embeddings, dim=0)
    return word_embeddings.contiguous().detach().cpu().numpy()


if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    phrase_id = pickle.load(open(pkl_dump_dir + "phrase_id_coarse_map.pkl", "rb"))
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    embeddings = pickle.load(open(pkl_dump_dir + "bert_word_embeddings.pkl", "rb"))

    phrases = list(phrase_id.keys())
    filtered_phrases = list(set(phrases) - set(embeddings.keys()))
    print("Number of Unknown phrases: ", len(filtered_phrases))

    none_phrase_count = 0
    for ph in filtered_phrases:
        temp = get_phrase_embedding(ph, embeddings, translator)
        if temp is None:
            none_phrase_count += 1
            embeddings[ph] = bert_embedding(model, tokenizer, ph)
            continue
        embeddings[ph] = temp

    print("None Phrases: ", none_phrase_count)

    for p in parent_to_child:
        for ch in parent_to_child[p]:
            child_label_str = " ".join([t for t in ch.split("_") if t not in stop_words]).strip()
            if child_label_str not in embeddings:
                temp = get_phrase_embedding(child_label_str, embeddings, translator)
                if temp is None:
                    embeddings[child_label_str] = bert_embedding(model, tokenizer, child_label_str)
                    continue
                embeddings[child_label_str] = temp

    pickle.dump(embeddings, open(pkl_dump_dir + "bert_word_phrase_embeddings.pkl", "wb"))
