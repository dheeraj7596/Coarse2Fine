import sys
import torch
import pickle
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup, AdamW
from sklearn.metrics import classification_report
from gpt2_coarse_finetune import gpt2_tokenize, test_generate, create_data_loaders, format_time
from torch.utils.data import TensorDataset
import numpy as np
import random
import time


def gpt2_fine_tokenize(tokenizer, df, index_to_label, pad_token_dict, max_length=768):
    input_ids = []
    attention_masks = []
    # For every sentence...
    sentences = df.text.values
    num_sentences = len(sentences)

    for i, sent in enumerate(sentences):
        sibling_input_ids = []
        sibling_attn_masks = []
        for ind_l in index_to_label:
            label = index_to_label[ind_l]
            temp_list = ["<|labelpad|>"] * pad_token_dict[label]
            if len(temp_list) > 0:
                label_str = label + " " + " ".join(temp_list)
            else:
                label_str = label
            encoded_dict = tokenizer.encode_plus(
                label_str + " <|labelsep|> " + sent,  # Sentence to encode.
                truncation=True,
                max_length=max_length - 2,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )
            encoded_dict['input_ids'] = torch.tensor(
                [[tokenizer.bos_token_id] + encoded_dict['input_ids'].data.tolist()[0] + [tokenizer.eos_token_id]]
            )
            encoded_dict['attention_mask'] = torch.tensor(
                [[1] + encoded_dict['attention_mask'].data.tolist()[0] + [1]]
            )
            sibling_input_ids.append(encoded_dict['input_ids'])
            sibling_attn_masks.append(encoded_dict['attention_mask'])

        # Add the encoded sentence to the list.
        input_ids.append(torch.cat(sibling_input_ids, dim=0))

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(torch.cat(sibling_attn_masks, dim=0))
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0).view(num_sentences, -1, max_length)
    attention_masks = torch.cat(attention_masks, dim=0).view(num_sentences, -1, max_length)

    return input_ids, attention_masks


def train(coarse_model, fine_model, train_dataloader, validation_dataloader, doc_start_ind, index_to_label, device):
    epsilon = 1e-40 # Defined to avoid log probability getting undefined.
    fine_posterior = torch.nn.Parameter(torch.ones(len(index_to_label)).to(device))
    optimizer = AdamW(list(fine_model.parameters()) + [fine_posterior],
                      lr=5e-4,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    criterion = torch.nn.KLDivLoss(size_average=False)
    sample_every = 100
    warmup_steps = 1e2
    epochs = 5
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []
    total_t0 = time.time()

    coarse_model.eval()

    for epoch_i in range(0, epochs):
        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)
        t0 = time.time()
        total_train_loss = 0
        fine_model.train()

        for step, batch in enumerate(train_dataloader):
            # batch contains -> coarse_input_ids, coarse_attention_masks, fine_input_ids, fine_attention_masks
            # todo write what to do if step%sample_every case
            fine_posterior_probs = torch.softmax(fine_posterior, dim=0)
            print(fine_posterior_probs)

            b_coarse_input_ids = batch[0].to(device)
            b_coarse_labels = batch[0].to(device)
            b_coarse_input_mask = batch[1].to(device)

            b_size = b_coarse_input_ids.shape[0]

            b_fine_input_ids_minibatch = batch[2].to(device)
            b_fine_input_mask_minibatch = batch[3].to(device)

            outputs = coarse_model(b_coarse_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_coarse_input_mask,
                                   labels=b_coarse_labels)

            coarse_logits = torch.softmax(outputs[1], dim=-1)[:, doc_start_ind:, :]
            batch_coarse_probs = coarse_logits.gather(2, b_coarse_labels[:, doc_start_ind:].unsqueeze(dim=-1)).squeeze(
                dim=-1)

            batch_fine_probs = []
            for b_ind in range(b_size):
                temp = 0
                for l_ind in index_to_label:
                    b_fine_input_ids = b_fine_input_ids_minibatch[b_ind, l_ind, :].unsqueeze(0).to(device)
                    b_fine_labels = b_fine_input_ids_minibatch[b_ind, l_ind, :].unsqueeze(0).to(device)
                    b_fine_input_mask = b_fine_input_mask_minibatch[b_ind, l_ind, :].unsqueeze(0).to(device)

                    outputs = fine_model(b_fine_input_ids,
                                         token_type_ids=None,
                                         attention_mask=b_fine_input_mask,
                                         labels=b_fine_labels)
                    fine_logits = torch.softmax(outputs[1], dim=-1)[:, doc_start_ind:, :]
                    fine_probs = fine_logits.gather(2, b_fine_labels[:, doc_start_ind:].unsqueeze(dim=-1)).squeeze(
                        dim=-1).squeeze(dim=0)
                    temp += fine_probs * fine_posterior_probs[l_ind]
                batch_fine_probs.append(temp + epsilon)

            batch_fine_probs = torch.cat(batch_fine_probs, dim=0)

            loss = criterion(batch_fine_probs.log(), batch_coarse_probs.detach())
            total_train_loss += loss.item()
            print("Loss:", loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epoch took: {:}".format(training_time), flush=True)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("", flush=True)
        print("Running Validation...", flush=True)

        t0 = time.time()

        fine_model.eval()

        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # batch contains -> coarse_input_ids, coarse_attention_masks, fine_input_ids, fine_attention_masks
            b_coarse_input_ids = batch[0].to(device)
            b_coarse_labels = batch[0].to(device)
            b_coarse_input_mask = batch[1].to(device)

            b_size = b_coarse_input_ids.shape[0]

            b_fine_input_ids_minibatch = batch[2].to(device)
            b_fine_input_mask_minibatch = batch[3].to(device)

            with torch.no_grad():
                fine_posterior_probs = torch.softmax(fine_posterior, dim=0)
                outputs = coarse_model(b_coarse_input_ids,
                                       token_type_ids=None,
                                       attention_mask=b_coarse_input_mask,
                                       labels=b_coarse_labels)

                coarse_logits = torch.softmax(outputs[1], dim=-1)[:, doc_start_ind:, :]
                batch_coarse_probs = coarse_logits.gather(2,
                                                          b_coarse_labels[:, doc_start_ind:].unsqueeze(dim=-1)).squeeze(
                    dim=-1)

                batch_fine_probs = []
                for b_ind in range(b_size):
                    temp = 0
                    for l_ind in index_to_label:
                        b_fine_input_ids = b_fine_input_ids_minibatch[b_ind, l_ind, :].unsqueeze(0).to(device)
                        b_fine_labels = b_fine_input_ids_minibatch[b_ind, l_ind, :].unsqueeze(0).to(device)
                        b_fine_input_mask = b_fine_input_mask_minibatch[b_ind, l_ind, :].unsqueeze(0).to(device)

                        outputs = fine_model(b_fine_input_ids,
                                             token_type_ids=None,
                                             attention_mask=b_fine_input_mask,
                                             labels=b_fine_labels)
                        fine_logits = torch.softmax(outputs[1], dim=-1)[:, doc_start_ind:, :]
                        fine_probs = fine_logits.gather(2, b_fine_labels[:, doc_start_ind:].unsqueeze(dim=-1)).squeeze(
                            dim=-1).squeeze(dim=0)
                        temp += fine_probs * fine_posterior_probs[l_ind]
                    batch_fine_probs.append(temp)

                batch_fine_probs = torch.cat(batch_fine_probs, dim=0)

            # Accumulate the validation loss.
            loss = criterion(batch_fine_probs.log(), batch_coarse_probs.detach())
            total_eval_loss += loss.item()

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss), flush=True)
        print("  Validation took: {:}".format(validation_time), flush=True)

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("", flush=True)
    print("Training complete!", flush=True)

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)), flush=True)
    return fine_posterior, fine_model


def create_pad_token_dict(p, parent_to_child, coarse_tokenizer, fine_tokenizer):
    pad_token_dict = {}
    children = parent_to_child[p]
    parent_tokens = coarse_tokenizer.tokenize(p)
    max_num = -1
    for ch in children:
        max_num = max(len(fine_tokenizer.tokenize(" ".join(ch.split("_")))), max_num)
    if len(parent_tokens) >= max_num:
        pad_token_dict[p] = 0
    else:
        pad_token_dict[p] = max_num - len(parent_tokens)
    for ch in children:
        ch_tokens = len(fine_tokenizer.tokenize(" ".join(ch.split("_"))))
        if ch_tokens >= max_num:
            pad_token_dict[ch] = 0
        else:
            pad_token_dict[ch] = max_num - ch_tokens
    doc_start_ind = 1 + max_num + 1  # this gives the token from which the document starts in the inputids, 1 for the starttoken, max_num for label infor, 1 for label_sup
    return doc_start_ind, pad_token_dict


# def test(fine_model, fine_posterior, test_dataloader, doc_start_ind, index_to_label, device):


if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    coarse_tok_path = pkl_dump_dir + "gpt2/tokenizer_coarse"
    model_path = pkl_dump_dir + "gpt2/model/"
    model_name = "coarse.pt"

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

    sibling_map = {}
    for p in parent_to_child:
        children = parent_to_child[p]
        for ch in children:
            sibling_map[ch] = [l for l in children]

    fine_labels = list(set(df.label.values))

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

    all_true = []
    all_preds = []
    for p in parent_to_child:
        print("Training coarse label:", p)
        fine_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>',
                                                       pad_token='<|pad|>',
                                                       additional_special_tokens=['<|labelsep|>', '<|labelpad|>'])
        fine_model = GPT2LMHeadModel.from_pretrained('gpt2')
        fine_model.resize_token_embeddings(len(fine_tokenizer))
        fine_model.to(device)

        children = parent_to_child[p]
        label_to_index = {}
        index_to_label = {}
        for i, l in enumerate(list(children)):
            label_to_index[l] = i
            index_to_label[i] = l

        doc_start_ind, pad_token_dict = create_pad_token_dict(p, parent_to_child, coarse_tokenizer, fine_tokenizer)

        temp_df = df[df.label.isin(children)].reset_index(drop=True)
        coarse_input_ids, coarse_attention_masks = gpt2_tokenize(coarse_tokenizer, temp_df, pad_token_dict)
        fine_input_ids, fine_attention_masks = gpt2_fine_tokenize(fine_tokenizer, temp_df, index_to_label,
                                                                  pad_token_dict)
        dataset = TensorDataset(coarse_input_ids, coarse_attention_masks, fine_input_ids, fine_attention_masks)

        train_dataloader, validation_dataloader = create_data_loaders(dataset, batch_size=1)
        fine_posterior, fine_model = train(coarse_model,
                                           fine_model,
                                           train_dataloader,
                                           validation_dataloader,
                                           doc_start_ind,
                                           index_to_label,
                                           device)
        test_generate(fine_model, fine_tokenizer, children, pad_token_dict, device)
        true, preds = test(fine_model, fine_posterior, test_dataloader, doc_start_ind, index_to_label, device)
        all_true += true
        all_preds += preds

        # todo save these at right location
        fine_tokenizer.save_pretrained(tok_path)
        torch.save(fine_model, model_path + model_name)

    print(classification_report(all_true, all_preds), flush=True)
    print("*" * 80, flush=True)
