import json
import torch
from transformers import BertTokenizer
import sys, pickle
from bert_glove import create_label_embeddings, test
from sklearn.metrics import classification_report

if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset
    # glove_dir = "/Users/dheerajmekala/Work/metaguide/data/glove.6B"
    glove_dir = "/data4/dheeraj/metaguide/glove.6B"

    tok_path = pkl_dump_dir + "bert/tokenizer_coarse"
    model_path = pkl_dump_dir + "bert/model/"
    model_name = "bert_vmf_coarse.pt"

    use_gpu = int(sys.argv[1])
    # use_gpu = False
    gpu_id = int(sys.argv[2])

    # Tell pytorch to run this model on the GPU.
    if use_gpu:
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device("cpu")

    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    parent_labels = list(parent_to_child.keys())

    df = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))
    label_set = set(df.label.values)
    label_to_index = {}
    index_to_label = {}
    for i, l in enumerate(list(label_set) + parent_labels):
        label_to_index[l] = i
        index_to_label[i] = l

    tokenizer = BertTokenizer.from_pretrained(tok_path, do_lower_case=True)
    model = torch.load(model_path + model_name)

    label_word_map = json.load(open(pkl_dump_dir + "label_word_map_coarse_fine.json", "r"))
    label_embeddings = create_label_embeddings(glove_dir, index_to_label, device, label_word_map)

    true = []
    preds = []
    for p in parent_labels:
        children = parent_to_child[p]
        temp_df = df[df.label.isin(children)].reset_index(drop=True)
        temp_parent_child = {label_to_index[p]: [label_to_index[ch] for ch in children]}
        temp_true, temp_preds = test(temp_df, tokenizer, model, label_embeddings, device, label_to_index,
                                     index_to_label, temp_parent_child)
        true += temp_true
        preds += temp_preds

    print(classification_report(true, preds))
