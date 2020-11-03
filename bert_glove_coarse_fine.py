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
from bert_glove import train, test, bert_tokenize, create_data_loaders, create_label_embeddings

if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset
    # glove_dir = "/Users/dheerajmekala/Work/metaguide/data/glove.6B"
    glove_dir = "/data4/dheeraj/metaguide/glove.6B"

    tok_path = pkl_dump_dir + "bert/tokenizer_coarse_fine"
    model_path = pkl_dump_dir + "bert/model/"
    model_name = "bert_vmf_coarse_fine.pt"

    os.makedirs(tok_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    use_gpu = int(sys.argv[1])
    # use_gpu = False
    gpu_id = int(sys.argv[2])

    # Tell pytorch to run this model on the GPU.
    if use_gpu:
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device("cpu")

    df = pickle.load(open(pkl_dump_dir + "df_coarse.pkl", "rb"))
    df_fine = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))

    df_train, df_test = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)

    # Tokenize all of the sentences and map the tokens to their word IDs.
    print('Loading BERT tokenizer...', flush=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    label_set = set(df_train.label.values).union(set(df_fine.label.values))
    label_to_index = {}
    index_to_label = {}
    for i, l in enumerate(list(label_set)):
        label_to_index[l] = i
        index_to_label[i] = l

    parent_child = {}
    for p in parent_to_child:
        parent_child[label_to_index[p]] = []
        for ch in parent_to_child[p]:
            parent_child[label_to_index[p]].append(label_to_index[ch])

    label_word_map = json.load(open(pkl_dump_dir + "label_word_map_coarse_fine.json", "r"))
    label_embeddings = create_label_embeddings(glove_dir, index_to_label, device, label_word_map)

    input_ids, attention_masks, labels = bert_tokenize(tokenizer, df_train, label_to_index)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-validation split.
    train_dataloader, validation_dataloader = create_data_loaders(dataset)

    model = BERTClass()
    model.to(device)

    model = train(train_dataloader, validation_dataloader, model, label_embeddings, device, epochs=5,
                  parent_child=parent_child)

    temp_parent_child = {}
    for p in parent_child:
        temp_parent_child[p] = p

    true, preds = test(df_test, tokenizer, model, label_embeddings, device, label_to_index, index_to_label,
                       temp_parent_child)
    print(classification_report(true, preds), flush=True)

    plot_confusion_mat(df_test["label"], preds, list(label_set))
    plt.savefig("./conf_mat.png")

    true, preds = test(df_train, tokenizer, model, label_embeddings, device, label_to_index, index_to_label,
                       temp_parent_child)
    print(classification_report(true, preds), flush=True)

    tokenizer.save_pretrained(tok_path)
    torch.save(model, model_path + model_name)
