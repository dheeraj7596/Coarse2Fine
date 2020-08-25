from transformers import BertTokenizer, BertModel
import pickle
import os
import torch
from classifier_seedwords import preprocess_df
import numpy as np
from nltk.corpus import stopwords
import string

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def get_phrase_bert_embeddings(embeddings, count, model, tokenizer, filtered_phrases, sentences):
    def tensor_to_numpy(tensor):
        return tensor.clone().detach().cpu().numpy()

    def find_sub_list(sl, l):
        results = []
        sll = len(sl)
        for ind in (i for i, e in enumerate(l) if e == sl[0]):
            if l[ind:ind + sll] == sl:
                results.append((ind, ind + sll))
        return results

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

        tokenized_to_phrase_id_indices = []
        present_phrases = []
        for ph in filtered_phrases:
            res = find_sub_list(ph.strip().split(), tokenized_text)
            for (start, end) in res:
                present_phrases.append(ph)
                try:
                    tokenized_to_phrase_id_indices.append(
                        (tokenized_to_id_indicies[start][0], tokenized_to_id_indicies[end][0]))
                except:
                    tokenized_to_phrase_id_indices.append(
                        (tokenized_to_id_indicies[start][0], tokenized_to_id_indicies[end - 1][1] + 1))

        if len(present_phrases) == 0:
            continue

        tokens_id = tokenizer.encode(" ".join(tokenized_text), add_special_tokens=True)
        assert tokens_id[1: -1] == check_tokens_id_list
        input_ids = torch.tensor([tokens_id], device=model.device)
        with torch.no_grad():
            hidden_states = model(input_ids)
        all_layer_outputs = hidden_states[2]
        last_layer = tensor_to_numpy(all_layer_outputs[layer].squeeze(0))[1: -1]

        for text, (start_index, end_index) in zip(present_phrases, tokenized_to_phrase_id_indices):
            phrase_vec = np.average(last_layer[start_index: end_index], axis=0)
            try:
                count[text] += 1
                embeddings[text] += phrase_vec
            except:
                count[text] = 1
                embeddings[text] = phrase_vec

    return embeddings, count


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

    phrase_id = pickle.load(open(pkl_dump_dir + "phrase_id_coarse_map.pkl", "rb"))
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    label_embeddings = pickle.load(open(pkl_dump_dir + "label_bert_word_embeddings.pkl", "rb"))

    phrases = list(phrase_id.keys())

    for p in parent_to_child:
        embeddings = label_embeddings[p]
        temp_df = df[df.label.isin([p])]
        temp_df = temp_df.reset_index(drop=True)

        filtered_phrases = list(set(phrases) - set(embeddings.keys()))

        for p in parent_to_child:
            for ch in parent_to_child[p]:
                child_label_str = " ".join([t for t in ch.split("_") if t not in stop_words]).strip()
                if child_label_str not in embeddings:
                    filtered_phrases.append(child_label_str)

        filtered_phrases = [ph.translate(translator) for ph in filtered_phrases]
        print("Number of Unknown phrases: ", len(filtered_phrases))

        count = {}
        embeddings, count = get_phrase_bert_embeddings(embeddings, count, model, tokenizer, filtered_phrases,
                                                       list(temp_df.text))

        for w in embeddings:
            try:
                embeddings[w] = embeddings[w] / count[w]
            except:
                continue

        label_embeddings[p] = embeddings

    pickle.dump(label_embeddings, open(pkl_dump_dir + "label_bert_word_phrase_embeddings.pkl", "wb"))
