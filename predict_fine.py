import json
import torch
from transformers import BertTokenizer
import sys, pickle
from bert_glove import create_label_embeddings, test

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

    df = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))
    label_set = set(df.label.values)
    label_to_index = {}
    index_to_label = {}
    for i, l in enumerate(list(label_set)):
        label_to_index[l] = i
        index_to_label[i] = l

    tokenizer = BertTokenizer.from_pretrained(tok_path, do_lower_case=True)
    model = torch.load(model_path + model_name)

    label_word_map = json.load(open(pkl_dump_dir + "label_word_map.json", "r"))
    label_embeddings = create_label_embeddings(glove_dir, index_to_label, device, label_word_map)

    preds = test(df, tokenizer, model, label_embeddings, device, index_to_label)
