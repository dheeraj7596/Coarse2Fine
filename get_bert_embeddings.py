import json
from transformers import BertTokenizer, BertModel
import pickle
import os
import torch
from classifier_seedwords import preprocess_df
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def get_zihan_bert_embeddings(embeddings, count, model, tokenizer, sentences):
    def tensor_to_numpy(tensor):
        return tensor.clone().detach().cpu().numpy()

    max_tokens = 512 - 2
    layer = 12
    for sentence in sentences:
        tokenized_text = tokenizer.basic_tokenizer.tokenize(sentence, never_split=tokenizer.all_special_tokens)
        _tokenized_text = []
        tokenized_to_id_indicies = []
        check_tokens_id_list = []
        cur_id_len = 0
        for index, token in enumerate(tokenized_text):
            tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
            if cur_id_len + len(tokens) <= max_tokens:
                _tokenized_text.append(token)
                tokenized_to_id_indicies.append((cur_id_len, cur_id_len + len(tokens)))
                cur_id_len += len(tokens)
                check_tokens_id_list.extend(tokenizer.convert_tokens_to_ids(tokens))
            else:
                break
        tokenized_text = _tokenized_text
        del _tokenized_text
        tokens_id = tokenizer.encode(" ".join(tokenized_text), add_special_tokens=True)
        assert tokens_id[1: -1] == check_tokens_id_list
        input_ids = torch.tensor([tokens_id], device=model.device)
        with torch.no_grad():
            hidden_states = model(input_ids)
        all_layer_outputs = hidden_states[2]
        last_layer = tensor_to_numpy(all_layer_outputs[layer].squeeze(0))[1: -1]
        for text, (start_index, end_index) in zip(tokenized_text, tokenized_to_id_indicies):
            word_vec = np.average(last_layer[start_index: end_index], axis=0)
            try:
                count[text] += 1
                embeddings[text] += word_vec
            except:
                count[text] = 1
                embeddings[text] = word_vec

    return embeddings, count


def get_bert_embeddings(embeddings, count, model, tokenizer, sentences):
    # concatenate last 4 layers for the word embedding
    batch_word_ids = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_word_ids = batch_word_ids.to(device)
    out = model(**batch_word_ids)
    hidden_states = out[2]  # tuple of size num_attn heads = 13, each containing batch_size x num_tokens x dim
    batch_embeddings = torch.stack(hidden_states, dim=0)  # (13 x batch_size x num_tokens x dim)
    batch_embeddings = batch_embeddings.permute(1, 2, 0, 3)  # (batch_size x num_tokens x 13 x dim)
    for i, sent in enumerate(sentences):
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
    return embeddings, count


if __name__ == "__main__":
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]


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
    embeddings = {}
    count = {}

    i = 0
    for batch in chunks(list(df.text), n=50):
        print("Batch: ", i)
        embeddings, count = get_zihan_bert_embeddings(embeddings, count, model, tokenizer, batch)
        i += 1
    pickle.dump(embeddings, open(pkl_dump_dir + "bert_word_embeddings.pkl", "wb"))
