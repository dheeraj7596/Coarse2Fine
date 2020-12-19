import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup, AdamW
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
import os
import pickle
import sys
import numpy as np
import random
import time
import datetime
import copy


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def basic_gpt2_tokenize(tokenizer, sentences, labels, pad_token_dict, max_length=768):
    input_ids = []
    attention_masks = []
    # For every sentence...
    for i, sent in enumerate(sentences):
        label = labels[i]
        temp_list = ["<|labelpad|>"] * pad_token_dict[label]
        if len(temp_list) > 0:
            label_str = " ".join(label.split("_")) + " " + " ".join(temp_list)
        else:
            label_str = " ".join(label.split("_"))
        encoded_dict = tokenizer.encode_plus(
            tokenizer.bos_token + " " + label_str + " <|labelsep|> " + sent,  # Sentence to encode.
            truncation=True,
            max_length=max_length,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


def gpt2_tokenize(tokenizer, sentences, label_strs, pad_token_dict, label_to_index, max_length=768):
    input_ids = []
    attention_masks = []
    # For every sentence...
    labels = copy.deepcopy(label_strs)
    for i, l in enumerate(list(labels)):
        labels[i] = label_to_index[l]
    labels = np.array(labels, dtype='int32')

    for i, sent in enumerate(sentences):
        label = label_strs[i]
        temp_list = ["<|labelpad|>"] * pad_token_dict[label]
        if len(temp_list) > 0:
            label_str = " ".join(label.split("_")) + " " + " ".join(temp_list)
        else:
            label_str = " ".join(label.split("_"))
        encoded_dict = tokenizer.encode_plus(
            tokenizer.bos_token + " " + label_str + " <|labelsep|> " + sent,  # Sentence to encode.
            truncation=True,
            max_length=max_length,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.LongTensor(labels)

    return input_ids, attention_masks, labels


def create_data_loaders(dataset, batch_size):
    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # The DataLoader needs to know our batch size for training, so we specify it
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch
    # size of 16 or 32.
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )
    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )
    return train_dataloader, validation_dataloader


def train(model, tokenizer, train_dataloader, validation_dataloader, index_to_label, pad_token_dict, doc_start_ind_dict,
          device):
    def calculate_loss(lm_logits, b_labels, b_input_mask, cls_labels, index_to_label, doc_start_ind_dict, loss_fct):
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

    optimizer = AdamW(model.parameters(),
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
        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % sample_every == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),
                      flush=True)
                model.eval()
                lbl = random.choice(list(index_to_label.values()))
                temp_list = ["<|labelpad|>"] * pad_token_dict[lbl]
                if len(temp_list) > 0:
                    label_str = " ".join(lbl.split("_")) + " " + " ".join(temp_list)
                else:
                    label_str = " ".join(lbl.split("_"))
                text = tokenizer.bos_token + " " + label_str + " <|labelsep|> "
                sample_outputs = model.generate(
                    input_ids=tokenizer.encode(text, return_tensors='pt').to(device),
                    do_sample=True,
                    top_k=50,
                    max_length=200,
                    top_p=0.95,
                    num_return_sequences=1
                )
                for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, tokenizer.decode(sample_output)), flush=True)
                model.train()

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            cls_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            loss = calculate_loss(outputs[1], b_labels, b_input_mask, cls_labels, index_to_label, doc_start_ind_dict,
                                  loss_fct)
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

        model.eval()

        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[0].to(device)
            cls_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)

            # Accumulate the validation loss.
            loss = calculate_loss(outputs[1], b_labels, b_input_mask, cls_labels, index_to_label, doc_start_ind_dict,
                                  loss_fct)
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
    return model


def test_generate(model, tokenizer, label_set, pad_token_dict, device):
    model.eval()
    for l in label_set:
        print("Generating sentence for label", l)
        temp_list = ["<|labelpad|>"] * pad_token_dict[l]
        if len(temp_list) > 0:
            label_str = " ".join(l.split("_")) + " " + " ".join(temp_list)
        else:
            label_str = " ".join(l.split("_"))
        text = tokenizer.bos_token + " " + label_str + " <|labelsep|> "
        sample_outputs = model.generate(
            input_ids=tokenizer.encode(text, return_tensors='pt').to(device),
            do_sample=True,
            top_k=50,
            max_length=200,
            top_p=0.95,
            num_return_sequences=1
        )
        for i, sample_output in enumerate(sample_outputs):
            print("{}: {}".format(i, tokenizer.decode(sample_output)))


if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    tok_path = pkl_dump_dir + "gpt2/tokenizer_coarse"
    model_path = pkl_dump_dir + "gpt2/model/"
    model_name = "coarse.pt"

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
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', pad_token='<|pad|>',
                                              additional_special_tokens=['<|labelsep|>', '<|labelpad|>'])

    pad_token_dict = {}
    doc_start_ind_dict = {}
    for p in parent_to_child:
        children = parent_to_child[p]
        parent_tokens = tokenizer.tokenize(" " + p)
        max_num = len(parent_tokens)
        for ch in children:
            max_num = max(len(tokenizer.tokenize(" " + " ".join(ch.split("_")))), max_num)
        pad_token_dict[p] = max_num - len(parent_tokens)
        doc_start_ind_dict[p] = 1 + max_num + 1
        # this gives the token from which the document starts in the inputids, 1 for the starttoken, max_num for label infor, 1 for label_sup

    # tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<|startoftext|> sports <|labelsep|> Hello, my dog is cute <|endoftext|>")))

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    label_set = set(df.label.values)
    label_to_index = {}
    index_to_label = {}
    for i, l in enumerate(list(label_set)):
        label_to_index[l] = i
        index_to_label[i] = l

    input_ids, attention_masks, labels = gpt2_tokenize(tokenizer, df.text.values, df.label.values, pad_token_dict,
                                                       label_to_index)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-validation split.
    train_dataloader, validation_dataloader = create_data_loaders(dataset, batch_size=4)

    model = train(model, tokenizer, train_dataloader, validation_dataloader, index_to_label, pad_token_dict,
                  doc_start_ind_dict, device)
    test_generate(model, tokenizer, label_set, pad_token_dict, device)

    tokenizer.save_pretrained(tok_path)
    torch.save(model, model_path + model_name)
    pickle.dump(pad_token_dict, open(pkl_dump_dir + "pad_token_dict.pkl", "wb"))
