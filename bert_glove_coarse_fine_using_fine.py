import json
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer
import os, sys, pickle
from bert_class import BERTClass
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from util import plot_confusion_mat
import matplotlib.pyplot as plt
from bert_glove import train, test, bert_tokenize, create_data_loaders


def order_label_embeddings(label_embeddings, index_to_label, label_word_map, device):
    mod_label_embeds = []
    for i in index_to_label:
        mod_label_embeds.append(label_embeddings[label_word_map[index_to_label[i]]])
    mod_label_embeds = torch.tensor(mod_label_embeds).to(device)
    return mod_label_embeds


def make_train_test(df, children, label_word_map):
    tokens = []
    for ch in children:
        tokens.append(label_word_map[ch])
    reg_exp = "|".join(tokens)

    df_train = df[df['text'].str.contains(reg_exp)].reset_index(drop=True)
    df_test = df[~df['text'].str.contains(reg_exp)].reset_index(drop=True)
    return df_train, df_test


if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    # tok_path = pkl_dump_dir + "bert/tokenizer_coarse_fine_using_fine"
    # model_path = pkl_dump_dir + "bert/model/"
    # model_name = "bert_vmf_coarse_fine_using_fine.pt"
    #
    # os.makedirs(tok_path, exist_ok=True)
    # os.makedirs(model_path, exist_ok=True)

    use_gpu = int(sys.argv[1])
    # use_gpu = False
    gpu_id = int(sys.argv[2])

    # Tell pytorch to run this model on the GPU.
    if use_gpu:
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device("cpu")

    df_fine = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    # df_train, df_test = train_test_split(df_fine, test_size=0.1, stratify=df_fine["label"], random_state=42)

    fine_labels = list(set(df_fine.label.values))

    label_set = fine_labels
    label_to_index = {}
    index_to_label = {}
    for i, l in enumerate(list(label_set)):
        label_to_index[l] = i
        index_to_label[i] = l

    fine_inds = []
    for f in fine_labels:
        fine_inds.append(label_to_index[f])

    label_word_map = json.load(open(pkl_dump_dir + "label_word_map_coarse_fine_using_fine.json", "r"))
    label_embeddings = pickle.load(open(pkl_dump_dir + "label_embeddings.pkl", "rb"))
    label_embeddings = order_label_embeddings(label_embeddings, index_to_label, label_word_map, device)

    # label_embeddings = create_label_embeddings(glove_dir, index_to_label, device, label_word_map)

    sibling_map = {}
    for p in parent_to_child:
        children = parent_to_child[p]
        for ch in children:
            sibling_map[label_to_index[ch]] = [label_to_index[l] for l in children]

    print('Loading BERT tokenizer...', flush=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    input_ids, attention_masks, labels = bert_tokenize(tokenizer, df_fine, label_to_index)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-validation split.
    train_dataloader, validation_dataloader = create_data_loaders(dataset)

    model = BERTClass()
    model.to(device)

    add_args = {}
    add_args["possible_labels"] = fine_inds
    add_args["contrastive_map"] = sibling_map

    model = train(train_dataloader, validation_dataloader, model, label_embeddings, device, epochs=5,
                  additional_args=add_args)

    print("****************** CLASSIFICATION REPORT ON All Data ********************", flush=True)

    all_true = []
    all_preds = []
    for p in parent_to_child:
        children = parent_to_child[p]
        temp_df = df_fine[df_fine.label.isin(children)].reset_index(drop=True)
        children_ids = [label_to_index[l] for l in children]

        test_add_args = {}
        test_add_args["possible_labels"] = children_ids
        test_add_args["contrastive_map"] = None

        true, preds = test(temp_df, tokenizer, model, label_embeddings, device, label_to_index, index_to_label,
                           test_add_args)
        all_true += true
        all_preds += preds
    print(classification_report(all_true, all_preds), flush=True)
    print("*" * 80, flush=True)
    # tokenizer.save_pretrained(tok_path)
    # torch.save(model, model_path + model_name)
