import torch
import sys
import pickle
from transformers import GPT2Tokenizer
from gpt2_fine_finetune import create_pad_token_dict, gpt2_fine_tokenize, test
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import copy
import numpy as np
from sklearn.metrics import classification_report

if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    base_fine_path = pkl_dump_dir + "gpt2/fine/"
    coarse_tok_path = pkl_dump_dir + "gpt2/tokenizer_coarse"

    coarse_tokenizer = GPT2Tokenizer.from_pretrained(coarse_tok_path, do_lower_case=True)

    use_gpu = int(sys.argv[1])
    # use_gpu = False
    gpu_id = int(sys.argv[2])

    # Tell pytorch to run this model on the GPU.
    if use_gpu:
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device("cpu")

    df = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))

    threshold = 0.7

    all_true = []
    all_preds = []
    all_sub_true = []
    all_sub_preds = []
    for p in parent_to_child:
        fine_label_path = base_fine_path + p
        fine_tok_path = fine_label_path + "/tokenizer"
        fine_model_path = fine_label_path + "/model/"

        fine_tokenizer = GPT2Tokenizer.from_pretrained(fine_tok_path, do_lower_case=True)
        fine_model = torch.load(fine_model_path + p + ".pt", map_location=device)
        fine_model.to(device)

        index_to_label = pickle.load(open(fine_label_path + "/index_to_label.pkl", "rb"))
        label_to_index = pickle.load(open(fine_label_path + "/label_to_index.pkl", "rb"))
        fine_posterior = torch.load(fine_label_path + "/fine_posterior.pt", map_location=device)

        children = parent_to_child[p]
        temp_df = df[df.label.isin(children)].reset_index(drop=True)
        doc_start_ind, pad_token_dict = create_pad_token_dict(p, parent_to_child, coarse_tokenizer, fine_tokenizer)
        fine_input_ids, fine_attention_masks = gpt2_fine_tokenize(fine_tokenizer, temp_df, index_to_label,
                                                                  pad_token_dict)

        true, preds, scores = test(fine_model, fine_posterior, fine_input_ids, fine_attention_masks, doc_start_ind,
                                   index_to_label, label_to_index, list(temp_df.label.values), device)
        all_true += true
        all_preds += preds

        probs = torch.softmax(scores, dim=-1)
        max_probs, max_inds = probs.max(dim=-1)
        b_size = probs.shape[0]

        sub_true = []
        sub_preds = []
        for i in range(b_size):
            if max_probs[i].item() >= threshold:
                sub_true.append(true[i])
                sub_preds.append(preds[i])

        print("Classification Report of True and Preds for", p)
        print(classification_report(true, preds), flush=True)
        print("*" * 80, flush=True)

        print("Classification Report of Sub_True and Sub_Preds for", p)
        print(classification_report(sub_true, sub_preds), flush=True)
        print("*" * 80, flush=True)

        all_sub_true += sub_true
        all_sub_preds += sub_preds

    print("Classification Report of All True and All Preds")
    print(classification_report(all_true, all_preds), flush=True)
    print("#" * 80, flush=True)
    print("Classification Report of All sub True and All sub Preds")
    print(classification_report(all_sub_true, all_sub_preds), flush=True)
    print("#" * 80, flush=True)
    # true_labels = list(temp_df.label.values)
    # # Set the batch size.
    # batch_size = 2
    # # Create the DataLoader.
    # labels = copy.deepcopy(true_labels)
    # for i, l in enumerate(list(labels)):
    #     labels[i] = label_to_index[l]
    # labels = np.array(labels, dtype='int32')
    # labels = torch.LongTensor(labels)
    #
    # prediction_data = TensorDataset(fine_input_ids, fine_attention_masks, labels)
    # prediction_sampler = SequentialSampler(prediction_data)
    # prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    #
    # # Tracking variables
    # predictions, true_labels = [], []
    #
    # fine_model.eval()
    # for batch in prediction_dataloader:
    #     # batch contains -> fine_input_ids, fine_attention_masks, fine_grained_labels
    #     b_fine_input_ids_minibatch = batch[0].to(device)
    #     b_fine_input_mask_minibatch = batch[1].to(device)
    #     b_cls_labels = batch[2].to(device)
    #     b_size = b_fine_input_ids_minibatch.shape[0]
    #
    #     with torch.no_grad():
    #         fine_posterior_log_probs = torch.log_softmax(fine_posterior, dim=0)
    #         batch_fine_logits = []
    #         for b_ind in range(b_size):
    #             label_log_probs = []
    #             for l_ind in index_to_label:
    #                 b_fine_input_ids = b_fine_input_ids_minibatch[b_ind, l_ind, :].unsqueeze(0).to(device)
    #                 b_fine_labels = b_fine_input_ids_minibatch[b_ind, l_ind, :].unsqueeze(0).to(device)
    #                 b_fine_input_mask = b_fine_input_mask_minibatch[b_ind, l_ind, :].unsqueeze(0).to(device)
    #
    #                 outputs = fine_model(b_fine_input_ids,
    #                                      token_type_ids=None,
    #                                      attention_mask=b_fine_input_mask,
    #                                      labels=b_fine_labels)
    #                 fine_logits = torch.log_softmax(outputs[1], dim=-1)[:, doc_start_ind:, :]
    #                 fine_log_probs = fine_logits.gather(2,
    #                                                     b_fine_labels[:, doc_start_ind:].unsqueeze(dim=-1)).squeeze(
    #                     dim=-1).squeeze(dim=0)
    #                 label_log_probs.append(fine_posterior_log_probs[l_ind] + fine_log_probs.sum())
    #             label_log_probs = torch.tensor(label_log_probs).unsqueeze(0)
    #             batch_fine_logits.append(label_log_probs)
    #
    #         batch_fine_logits = torch.cat(batch_fine_logits, dim=0)
    #
    #     predictions.append(batch_fine_logits.detach().cpu().numpy())
    #     label_ids = b_cls_labels.to('cpu').numpy()
    #     true_labels.append(label_ids)
    #
    # preds = []
    # for pred in predictions:
    #     preds = preds + list(pred.argmax(axis=-1))
    #
    # true = []
    # for t in true_labels:
    #     true = true + list(t)
    #
    # for i, t in enumerate(true):
    #     true[i] = index_to_label[t]
    #     preds[i] = index_to_label[preds[i]]
    #
    # print(classification_report(true, preds), flush=True)
