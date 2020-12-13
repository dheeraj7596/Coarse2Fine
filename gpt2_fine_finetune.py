import sys
import torch
import pickle
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup, AdamW
from sklearn.metrics import classification_report
from gpt2_coarse_finetune import gpt2_tokenize, test_generate, create_data_loaders, format_time
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data import TensorDataset
import numpy as np
import random
import time
import os
import copy


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
            processed_label_str = " ".join(label.split("_"))
            temp_list = ["<|labelpad|>"] * pad_token_dict[label]
            if len(temp_list) > 0:
                label_str = processed_label_str + " " + " ".join(temp_list)
            else:
                label_str = processed_label_str
            encoded_dict = tokenizer.encode_plus(
                label_str + " <|labelsep|> " + sent,  # Sentence to encode.
                truncation=True,
                max_length=max_length - 1,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )
            encoded_dict['input_ids'] = torch.tensor(
                [[tokenizer.bos_token_id] + encoded_dict['input_ids'].data.tolist()[0]]
            )
            encoded_dict['attention_mask'] = torch.tensor(
                [[1] + encoded_dict['attention_mask'].data.tolist()[0]]
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


def train(coarse_model, fine_model, coarse_tokenizer, fine_tokenizer, train_dataloader, validation_dataloader,
          doc_start_ind, index_to_label, device):
    def calculate_loss(batch_fine_probs, batch_coarse_probs, batch_fine_input_masks, batch_coarse_input_masks,
                       batch_fine_input_ids, batch_coarse_input_ids, coarse_tokenizer, fine_tokenizer,
                       doc_start_ind, loss_fct):
        # Remove pad tokens
        # consider from doc_start_ind - 1
        batch_size = batch_fine_probs.shape[0]
        losses = []
        for b in range(batch_size):
            fine_logits_ind = batch_fine_probs[b, :, :]  # seq_len x |V|
            coarse_logits_ind = batch_coarse_probs[b, :, :]  # seq_len x |V|
            fine_mask = batch_fine_input_masks[b, :] > 0
            coarse_mask = batch_coarse_input_masks[b, :] > 0
            if not torch.all(fine_mask.eq(coarse_mask)):
                print("Fine sentence", fine_tokenizer.decode(batch_fine_input_ids[b, :]))
                print("Coarse sentence", coarse_tokenizer.decode(batch_coarse_input_ids[b, :]))
                raise Exception("Fine and Coarse mask is not same")

            fine_maski = fine_mask.unsqueeze(-1).expand_as(fine_logits_ind)
            coarse_maski = coarse_mask.unsqueeze(-1).expand_as(coarse_logits_ind)
            # unpad_seq_len x |V|
            fine_logits_pad_removed = torch.masked_select(fine_logits_ind, fine_maski).view(-1,
                                                                                            fine_logits_ind.size(-1))
            coarse_logits_pad_removed = torch.masked_select(coarse_logits_ind, coarse_maski).view(-1,
                                                                                                  coarse_logits_ind.size(
                                                                                                      -1))
            shift_fine_logits = fine_logits_pad_removed[doc_start_ind - 1:-1, :].contiguous()
            shift_coarse_logits = coarse_logits_pad_removed[doc_start_ind - 1:-1, :].contiguous()
            # Compute loss here of shift_fine_logits and shift_coarse_logits append to losses
            loss = loss_fct(shift_fine_logits, shift_coarse_logits).unsqueeze(0)
            losses.append(loss)

        # Return mean of losses here
        losses = torch.cat(losses, dim=0)
        return losses.mean()

    # epsilon = 1e-20  # Defined to avoid log probability getting undefined.
    fine_posterior = torch.nn.Parameter(torch.ones(len(index_to_label)).to(device))
    optimizer = AdamW(list(fine_model.parameters()) + [fine_posterior],
                      lr=5e-4,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
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
            if step % sample_every == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),
                      flush=True)
                fine_model.eval()
                lbl = random.choice(list(index_to_label.values()))
                temp_list = ["<|labelpad|>"] * pad_token_dict[lbl]
                if len(temp_list) > 0:
                    label_str = " ".join(lbl.split("_")) + " " + " ".join(temp_list)
                else:
                    label_str = " ".join(lbl.split("_"))
                text = fine_tokenizer.bos_token + " " + label_str + " <|labelsep|> "
                sample_outputs = fine_model.generate(
                    input_ids=fine_tokenizer.encode(text, return_tensors='pt').to(device),
                    do_sample=True,
                    top_k=50,
                    max_length=200,
                    top_p=0.95,
                    num_return_sequences=1
                )
                for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, fine_tokenizer.decode(sample_output)), flush=True)
                fine_model.train()

            fine_posterior_log_probs = torch.log_softmax(fine_posterior, dim=0)
            print(torch.softmax(fine_posterior, dim=0), flush=True)

            b_coarse_input_ids = batch[0].to(device)
            b_coarse_labels = batch[0].to(device)
            b_coarse_input_mask = batch[1].to(device)

            b_size = b_coarse_input_ids.shape[0]

            b_fine_input_ids_minibatch = batch[2].to(device)
            b_fine_input_mask_minibatch = batch[3].to(device)

            coarse_model.zero_grad()
            fine_model.zero_grad()

            outputs = coarse_model(b_coarse_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_coarse_input_mask,
                                   labels=b_coarse_labels)

            batch_coarse_probs = torch.softmax(outputs[1], dim=-1)  # (b_size, seq_len, |V|)

            batch_fine_probs = []
            batch_fine_input_masks = []
            batch_fine_input_ids = []
            for b_ind in range(b_size):
                fine_label_sum_log_probs = []
                for l_ind in index_to_label:
                    b_fine_input_ids = b_fine_input_ids_minibatch[b_ind, l_ind, :].unsqueeze(0).to(device)
                    b_fine_labels = b_fine_input_ids_minibatch[b_ind, l_ind, :].unsqueeze(0).to(device)
                    b_fine_input_mask = b_fine_input_mask_minibatch[b_ind, l_ind, :].unsqueeze(0).to(device)

                    outputs = fine_model(b_fine_input_ids,
                                         token_type_ids=None,
                                         attention_mask=b_fine_input_mask,
                                         labels=b_fine_labels)
                    fine_log_probs = torch.log_softmax(outputs[1], dim=-1)
                    fine_label_sum_log_probs.append((fine_log_probs + fine_posterior_log_probs[l_ind]))

                fine_label_sum_log_probs = torch.cat(fine_label_sum_log_probs, dim=0)  # (|F|, seq_len, |V|)
                batch_fine_probs.append(fine_label_sum_log_probs.unsqueeze(0))
                batch_fine_input_ids.append(b_fine_input_ids)
                batch_fine_input_masks.append(b_fine_input_mask)

            batch_fine_probs = torch.cat(batch_fine_probs, dim=0)  # (b_size, |F|, seq_len, |V|)
            batch_fine_input_masks = torch.cat(batch_fine_input_masks, dim=0)  # (b_size, seq_len)
            batch_fine_input_ids = torch.cat(batch_fine_input_ids, dim=0)  # (b_size, seq_len)
            batch_fine_log_probs = torch.logsumexp(batch_fine_probs, dim=1)  # This computes logsum_i P(f_i|c) P(D|f_i)

            loss = calculate_loss(batch_fine_log_probs, batch_coarse_probs, batch_fine_input_masks, b_coarse_input_mask,
                                  batch_fine_input_ids, b_coarse_input_ids, coarse_tokenizer, fine_tokenizer,
                                  doc_start_ind, loss_fct)
            # loss = criterion(batch_fine_probs.log(), batch_coarse_probs.detach()).sum(dim=-1).mean(dim=-1).mean(dim=-1)
            total_train_loss += loss.item()
            print("Loss:", loss.item(), flush=True)

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
                fine_posterior_log_probs = torch.log_softmax(fine_posterior, dim=0)
                outputs = coarse_model(b_coarse_input_ids,
                                       token_type_ids=None,
                                       attention_mask=b_coarse_input_mask,
                                       labels=b_coarse_labels)

                batch_coarse_probs = torch.softmax(outputs[1], dim=-1)  # (b_size, seq_len, |V|)

                batch_fine_probs = []
                batch_fine_input_masks = []
                batch_fine_input_ids = []
                for b_ind in range(b_size):
                    fine_label_sum_log_probs = []
                    for l_ind in index_to_label:
                        b_fine_input_ids = b_fine_input_ids_minibatch[b_ind, l_ind, :].unsqueeze(0).to(device)
                        b_fine_labels = b_fine_input_ids_minibatch[b_ind, l_ind, :].unsqueeze(0).to(device)
                        b_fine_input_mask = b_fine_input_mask_minibatch[b_ind, l_ind, :].unsqueeze(0).to(device)

                        outputs = fine_model(b_fine_input_ids,
                                             token_type_ids=None,
                                             attention_mask=b_fine_input_mask,
                                             labels=b_fine_labels)
                        fine_log_probs = torch.log_softmax(outputs[1], dim=-1)
                        fine_label_sum_log_probs.append((fine_log_probs + fine_posterior_log_probs[l_ind]))

                    fine_label_sum_log_probs = torch.cat(fine_label_sum_log_probs, dim=0)  # (|F|, seq_len, |V|)
                    batch_fine_probs.append(fine_label_sum_log_probs.unsqueeze(0))
                    batch_fine_input_ids.append(b_fine_input_ids)
                    batch_fine_input_masks.append(b_fine_input_mask)

                batch_fine_probs = torch.cat(batch_fine_probs, dim=0)  # (b_size, |F|, seq_len, |V|)
                batch_fine_input_masks = torch.cat(batch_fine_input_masks, dim=0)  # (b_size, seq_len)
                batch_fine_input_ids = torch.cat(batch_fine_input_ids, dim=0)  # (b_size, seq_len)
                batch_fine_log_probs = torch.logsumexp(batch_fine_probs,
                                                       dim=1)  # This computes logsum_i P(f_i|c) P(D|f_i)

            # Accumulate the validation loss.
            loss = calculate_loss(batch_fine_log_probs, batch_coarse_probs, batch_fine_input_masks, b_coarse_input_mask,
                                  batch_fine_input_ids, b_coarse_input_ids, coarse_tokenizer, fine_tokenizer,
                                  doc_start_ind, loss_fct)
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
    max_num = len(parent_tokens)
    for ch in children:
        max_num = max(len(fine_tokenizer.tokenize(" ".join(ch.split("_")))), max_num)
    pad_token_dict[p] = max_num - len(parent_tokens)
    for ch in children:
        ch_tokens = len(fine_tokenizer.tokenize(" ".join(ch.split("_"))))
        pad_token_dict[ch] = max_num - ch_tokens
    doc_start_ind = 1 + max_num + 1  # this gives the token from which the document starts in the inputids, 1 for the starttoken, max_num for label infor, 1 for label_sup
    return doc_start_ind, pad_token_dict


def test(fine_model, fine_posterior, fine_input_ids, fine_attention_masks, doc_start_ind, index_to_label,
         label_to_index, true_labels, device):
    # Set the batch size.
    batch_size = 2
    # Create the DataLoader.
    labels = copy.deepcopy(true_labels)
    for i, l in enumerate(list(labels)):
        labels[i] = label_to_index[l]
    labels = np.array(labels, dtype='int32')
    labels = torch.LongTensor(labels)

    prediction_data = TensorDataset(fine_input_ids, fine_attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Tracking variables
    predictions, true_labels = [], []

    fine_model.eval()
    for batch in prediction_dataloader:
        # batch contains -> fine_input_ids, fine_attention_masks, fine_grained_labels
        b_fine_input_ids_minibatch = batch[0].to(device)
        b_fine_input_mask_minibatch = batch[1].to(device)
        b_cls_labels = batch[2].to(device)
        b_size = b_fine_input_ids_minibatch.shape[0]

        with torch.no_grad():
            fine_posterior_log_probs = torch.log_softmax(fine_posterior, dim=0)
            batch_fine_logits = []
            for b_ind in range(b_size):
                label_log_probs = []
                for l_ind in index_to_label:
                    b_fine_input_ids = b_fine_input_ids_minibatch[b_ind, l_ind, :].unsqueeze(0).to(device)
                    b_fine_labels = b_fine_input_ids_minibatch[b_ind, l_ind, :].unsqueeze(0).to(device)
                    b_fine_input_mask = b_fine_input_mask_minibatch[b_ind, l_ind, :].unsqueeze(0).to(device)

                    outputs = fine_model(b_fine_input_ids,
                                         token_type_ids=None,
                                         attention_mask=b_fine_input_mask,
                                         labels=b_fine_labels)
                    mask = b_fine_input_mask > 0

                    fine_logits = torch.log_softmax(outputs[1], dim=-1)
                    maski = mask.unsqueeze(-1).expand_as(fine_logits)
                    fine_logits_pad_removed = torch.masked_select(fine_logits, maski).view(-1, fine_logits.size(
                        -1)).unsqueeze(0)
                    fine_logits_pad_removed = fine_logits_pad_removed[:, doc_start_ind - 1:-1, :]

                    b_fine_labels_pad_removed = torch.masked_select(b_fine_labels, mask).unsqueeze(0)
                    b_fine_labels_pad_removed = b_fine_labels_pad_removed[:, doc_start_ind:]
                    fine_log_probs = fine_logits_pad_removed.gather(2, b_fine_labels_pad_removed.unsqueeze(
                        dim=-1)).squeeze(dim=-1).squeeze(dim=0)
                    label_log_probs.append(fine_posterior_log_probs[l_ind] + fine_log_probs.sum())
                label_log_probs = torch.tensor(label_log_probs).unsqueeze(0)
                batch_fine_logits.append(label_log_probs)

            batch_fine_logits = torch.cat(batch_fine_logits, dim=0)

        predictions.append(batch_fine_logits.detach().cpu().numpy())
        label_ids = b_cls_labels.to('cpu').numpy()
        true_labels.append(label_ids)

    preds = []
    for pred in predictions:
        preds = preds + list(pred.argmax(axis=-1))

    true = []
    for t in true_labels:
        true = true + list(t)

    for i, t in enumerate(true):
        true[i] = index_to_label[t]
        preds[i] = index_to_label[preds[i]]

    print(classification_report(true, preds), flush=True)
    return true, preds


if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    base_fine_path = pkl_dump_dir + "gpt2/fine/"

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
        fine_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', pad_token='<|pad|>',
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
        temp_coarse_lbls = [p] * len(temp_df.text.values)
        temp_coarse_label_to_index = {p: 0}

        coarse_input_ids, coarse_attention_masks, _ = gpt2_tokenize(coarse_tokenizer, temp_df.text.values,
                                                                    temp_coarse_lbls, pad_token_dict,
                                                                    temp_coarse_label_to_index)
        fine_input_ids, fine_attention_masks = gpt2_fine_tokenize(fine_tokenizer, temp_df, index_to_label,
                                                                  pad_token_dict)
        dataset = TensorDataset(coarse_input_ids, coarse_attention_masks, fine_input_ids, fine_attention_masks)

        train_dataloader, validation_dataloader = create_data_loaders(dataset, batch_size=1)
        fine_posterior, fine_model = train(coarse_model,
                                           fine_model,
                                           coarse_tokenizer,
                                           fine_tokenizer,
                                           train_dataloader,
                                           validation_dataloader,
                                           doc_start_ind,
                                           index_to_label,
                                           device)
        test_generate(fine_model, fine_tokenizer, children, pad_token_dict, device)
        true, preds = test(fine_model, fine_posterior, fine_input_ids, fine_attention_masks, doc_start_ind,
                           index_to_label, label_to_index, list(temp_df.label.values), device)
        all_true += true
        all_preds += preds

        fine_label_path = base_fine_path + p
        os.makedirs(fine_label_path, exist_ok=True)
        fine_tok_path = fine_label_path + "/tokenizer"
        fine_model_path = fine_label_path + "/model/"
        os.makedirs(fine_tok_path, exist_ok=True)
        os.makedirs(fine_model_path, exist_ok=True)

        fine_tokenizer.save_pretrained(fine_tok_path)
        torch.save(fine_model, fine_model_path + p + ".pt")
        torch.save(fine_posterior, fine_label_path + "/fine_posterior.pt")
        pickle.dump(index_to_label, open(fine_label_path + "/index_to_label.pkl", "wb"))
        pickle.dump(label_to_index, open(fine_label_path + "/label_to_index.pkl", "wb"))

        print("*" * 80)

    print(classification_report(all_true, all_preds), flush=True)
    print("*" * 80, flush=True)
