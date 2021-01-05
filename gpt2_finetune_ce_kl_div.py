import sys
import torch
import pickle
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from gpt2_coarse_finetune import gpt2_tokenize, test_generate, create_data_loaders, get_linear_schedule_with_warmup, \
    AdamW, CrossEntropyLoss, time, format_time
from gpt2_fine_finetune import create_pad_token_dict, gpt2_fine_tokenize
from gpt2_finetune_politics_prestart import func
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import TensorDataset
import numpy as np
import random
import os
import pandas as pd


def train(fine_model, fine_tokenizer, train_dataloader, validation_dataloader, kl_dataloader, index_to_label,
          pad_token_dict, doc_start_ind_dict, device, coarse_model, coarse_tokenizer, doc_start_ind):
    def calculate_ce_loss(lm_logits, b_labels, b_input_mask, cls_labels, index_to_label, doc_start_ind_dict, loss_fct):
        batch_size = lm_logits.shape[0]
        logits_collected = []
        labels_collected = []
        for b in range(batch_size):
            logits_ind = lm_logits[b, :, :]  # seq_len x |V|
            labels_ind = b_labels[b, :]  # seq_len
            mask = b_input_mask[b, :] > 0
            maski = mask.unsqueeze(-1).expand_as(logits_ind)
            # unpad_seq_len x |V|
            logits_pad_removed = torch.masked_select(logits_ind, maski).view(-1, logits_ind.size(-1))
            labels_pad_removed = torch.masked_select(labels_ind, mask)  # unpad_seq_len

            doc_start_ind = doc_start_ind_dict[index_to_label[cls_labels[b].item()]]
            shift_logits = logits_pad_removed[doc_start_ind - 1:-1, :].contiguous()
            shift_labels = labels_pad_removed[doc_start_ind:].contiguous()
            # Flatten the tokens
            logits_collected.append(shift_logits.view(-1, shift_logits.size(-1)))
            labels_collected.append(shift_labels.view(-1))

        logits_collected = torch.cat(logits_collected, dim=0)
        labels_collected = torch.cat(labels_collected, dim=0)
        loss = loss_fct(logits_collected, labels_collected)
        return loss

    def calculate_kl_div_loss(batch_fine_probs, batch_coarse_probs, batch_fine_input_masks, batch_coarse_input_masks,
                              batch_fine_input_ids, batch_coarse_input_ids, coarse_tokenizer, fine_tokenizer,
                              doc_start_ind):
        # Remove pad tokens
        # consider from doc_start_ind - 1
        loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
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

            fine_dec_sent = fine_tokenizer.decode(batch_fine_input_ids[b, :][doc_start_ind:])
            coarse_dec_sent = coarse_tokenizer.decode(batch_coarse_input_ids[b, :][doc_start_ind:])

            if fine_dec_sent != coarse_dec_sent:
                print("Fine sentence ", fine_tokenizer.decode(batch_fine_input_ids[b, :][doc_start_ind:]))
                print("Coarse sentence ", coarse_tokenizer.decode(batch_coarse_input_ids[b, :][doc_start_ind:]))
                raise Exception("Fine and Coarse decoded sentence is not same")

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

    def fwd_pass_kl_loss(fine_posterior, kl_dataloader, device, coarse_model, fine_model, coarse_tokenizer,
                         fine_tokenizer, doc_start_ind):
        it = 0
        for step, batch in kl_dataloader:
            fine_posterior_log_probs = torch.log_softmax(fine_posterior, dim=0)
            print(torch.softmax(fine_posterior, dim=0), flush=True)

            b_coarse_input_ids = batch[0].to(device)
            b_coarse_labels = batch[0].to(device)
            b_coarse_input_mask = batch[1].to(device)
            b_size = b_coarse_input_ids.shape[0]
            b_fine_input_ids_minibatch = batch[2].to(device)
            b_fine_input_mask_minibatch = batch[3].to(device)

            coarse_model.zero_grad()
            outputs = coarse_model(b_coarse_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_coarse_input_mask,
                                   labels=b_coarse_labels)
            batch_coarse_probs = torch.softmax(outputs[1], dim=-1).to(device)  # (b_size, seq_len, |V|)
            b_coarse_input_ids = b_coarse_input_ids.to(device)
            b_coarse_input_mask = b_coarse_input_mask.to(device)

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

            it += 1
            if it == 1:
                break

        batch_fine_probs = torch.cat(batch_fine_probs, dim=0)  # (b_size, |F|, seq_len, |V|)
        batch_fine_input_masks = torch.cat(batch_fine_input_masks, dim=0)  # (b_size, seq_len)
        batch_fine_input_ids = torch.cat(batch_fine_input_ids, dim=0)  # (b_size, seq_len)
        batch_fine_log_probs = torch.logsumexp(batch_fine_probs, dim=1)  # This computes logsum_i P(f_i|c) P(D|f_i)

        loss = calculate_kl_div_loss(batch_fine_log_probs, batch_coarse_probs, batch_fine_input_masks,
                                     b_coarse_input_mask, batch_fine_input_ids, b_coarse_input_ids,
                                     coarse_tokenizer, fine_tokenizer, doc_start_ind)
        return loss

    def calculate_loss(lm_logits, b_labels, b_input_mask, cls_labels, index_to_label, doc_start_ind_dict, loss_fct,
                       kl_dataloader, fine_posterior, device, coarse_model, fine_model, coarse_tokenizer,
                       fine_tokenizer, doc_start_ind, is_val=False):
        ce_loss = calculate_ce_loss(lm_logits, b_labels, b_input_mask, cls_labels, index_to_label, doc_start_ind_dict,
                                    loss_fct)
        if is_val:
            kl_loss = 0
            print("KL-loss", kl_loss, "CE-loss", ce_loss.item())
            return ce_loss
        else:
            kl_loss = fwd_pass_kl_loss(fine_posterior, kl_dataloader, device, coarse_model, fine_model,
                                       coarse_tokenizer, fine_tokenizer, doc_start_ind)
            print("KL-loss", kl_loss.item(), "CE-loss", ce_loss.item())
            return ce_loss + kl_loss

    fine_posterior = torch.nn.Parameter(torch.ones(len(index_to_label)).to(device))
    optimizer = AdamW(list(fine_model.parameters()) + [fine_posterior],
                      lr=5e-4,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )

    loss_fct = CrossEntropyLoss()
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

    for epoch_i in range(0, epochs):
        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)
        t0 = time.time()
        total_train_loss = 0
        fine_model.train()

        for step, batch in enumerate(train_dataloader):
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

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            cls_labels = batch[2].to(device)

            fine_model.zero_grad()

            outputs = fine_model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            loss = calculate_loss(outputs[1], b_labels, b_input_mask, cls_labels, index_to_label, doc_start_ind_dict,
                                  loss_fct, kl_dataloader, fine_posterior, device, coarse_model, fine_model,
                                  coarse_tokenizer, fine_tokenizer, doc_start_ind, is_val=False)

            # loss = outputs[0]
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epcoh took: {:}".format(training_time), flush=True)

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
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[0].to(device)
            cls_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = fine_model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)

            # Accumulate the validation loss.
            loss = calculate_loss(outputs[1], b_labels, b_input_mask, cls_labels, index_to_label, doc_start_ind_dict,
                                  loss_fct, kl_dataloader, fine_posterior, device, coarse_model, fine_model,
                                  coarse_tokenizer, fine_tokenizer, doc_start_ind, is_val=True)
            # loss = outputs[0]
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
    return fine_model


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

    df_fine = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))
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

        temp_df = df_fine[df_fine.label.isin(children)].reset_index(drop=True)
        temp_coarse_lbls = [p] * len(temp_df.text.values)
        temp_coarse_label_to_index = {p: 0}

        coarse_input_ids, coarse_attention_masks, _ = gpt2_tokenize(coarse_tokenizer, temp_df.text.values,
                                                                    temp_coarse_lbls, pad_token_dict,
                                                                    temp_coarse_label_to_index)
        fine_input_ids, fine_attention_masks = gpt2_fine_tokenize(fine_tokenizer, temp_df, index_to_label,
                                                                  pad_token_dict)

        kl_dataset = TensorDataset(coarse_input_ids, coarse_attention_masks, fine_input_ids, fine_attention_masks)
        kl_dataloader = DataLoader(
            kl_dataset,  # The training samples.
            sampler=RandomSampler(kl_dataset),  # Select batches randomly
            batch_size=1  # Trains with this batch size.
        )
        kl_dataloader = func(kl_dataloader)

        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_masks, labels)

        # Create a 90-10 train-validation split.
        train_dataloader, validation_dataloader = create_data_loaders(dataset, batch_size=4)

        doc_start_ind_dict = {}
        for ch in children:
            doc_start_ind_dict[ch] = doc_start_ind

        model = train(fine_model, fine_tokenizer, train_dataloader, validation_dataloader, kl_dataloader,
                      index_to_label, pad_token_dict, doc_start_ind_dict, device, coarse_model, coarse_tokenizer,
                      doc_start_ind)
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
