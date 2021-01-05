import sys
import torch
import pickle
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from gpt2_coarse_finetune import gpt2_tokenize, test_generate, create_data_loaders, train
from gpt2_fine_finetune import create_pad_token_dict
from torch.utils.data import TensorDataset
import numpy as np
import random
import os
import pandas as pd

if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    basepath = "/data4/dheeraj/coarse2fine/"
    dataset = sys.argv[6] + "/"
    pkl_dump_dir = basepath + dataset

    coarse_tok_path = pkl_dump_dir + "gpt2/tokenizer_coarse"
    model_path = pkl_dump_dir + "gpt2/model/"
    model_name = "coarse.pt"

    use_gpu = int(sys.argv[1])
    # use_gpu = False
    gpu_id = int(sys.argv[2])
    iteration = int(sys.argv[3])
    parent_label = sys.argv[4]
    fine_dir_name = sys.argv[5]

    base_fine_path = pkl_dump_dir + "gpt2/" + fine_dir_name + "/"

    # Tell pytorch to run this model on the GPU.
    if use_gpu:
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device("cpu")

    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))

    coarse_tokenizer = GPT2Tokenizer.from_pretrained(coarse_tok_path, do_lower_case=True)
    coarse_model = torch.load(model_path + model_name, map_location=device)
    coarse_model.to(device)

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for p in [parent_label]:
        print("Training coarse label:", p)
        fine_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', pad_token='<|pad|>',
                                                       additional_special_tokens=['<|labelsep|>', '<|labelpad|>'])
        fine_model = GPT2LMHeadModel.from_pretrained('gpt2')
        fine_model.resize_token_embeddings(len(fine_tokenizer))
        # fine_model = torch.nn.DataParallel(fine_model, device_ids=[1, 2])
        fine_model.to(device)

        children = parent_to_child[p]
        label_to_index = {}
        index_to_label = {}
        for i, l in enumerate(list(children)):
            label_to_index[l] = i
            index_to_label[i] = l
            if fine_dir_name == "fine":
                temp_df = pickle.load(open(pkl_dump_dir + "exclusive_" + str(iteration) + "it/" + l + ".pkl", "rb"))
            else:
                temp_df = pickle.load(
                    open(pkl_dump_dir + "exclusive_ceonly_" + str(iteration) + "it/" + l + ".pkl", "rb"))
            temp_df["label"] = [l] * len(temp_df)
            if i == 0:
                df = temp_df
            else:
                df = pd.concat([df, temp_df])

        doc_start_ind, pad_token_dict = create_pad_token_dict(p, parent_to_child, coarse_tokenizer, fine_tokenizer)
        print(pad_token_dict, doc_start_ind)

        input_ids, attention_masks, labels = gpt2_tokenize(fine_tokenizer, df.text.values, df.label.values,
                                                           pad_token_dict, label_to_index)

        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_masks, labels)

        # Create a 90-10 train-validation split.
        train_dataloader, validation_dataloader = create_data_loaders(dataset, batch_size=4)

        doc_start_ind_dict = {}
        for ch in children:
            doc_start_ind_dict[ch] = doc_start_ind

        model = train(fine_model, fine_tokenizer, train_dataloader, validation_dataloader, index_to_label,
                      pad_token_dict, doc_start_ind_dict, device)
        test_generate(model, fine_tokenizer, set(children), pad_token_dict, device)

        fine_label_path = base_fine_path + p
        os.makedirs(fine_label_path, exist_ok=True)
        fine_tok_path = fine_label_path + "/tokenizer"
        fine_model_path = fine_label_path + "/model/"
        os.makedirs(fine_tok_path, exist_ok=True)
        os.makedirs(fine_model_path, exist_ok=True)

        fine_tokenizer.save_pretrained(fine_tok_path)
        torch.save(fine_model, fine_model_path + p + ".pt")
        pickle.dump(index_to_label, open(fine_label_path + "/index_to_label.pkl", "wb"))
        pickle.dump(label_to_index, open(fine_label_path + "/label_to_index.pkl", "wb"))
        pickle.dump(pad_token_dict, open(fine_label_path + "/pad_token_dict.pkl", "wb"))

        print("*" * 80)
