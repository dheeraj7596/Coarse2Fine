import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import random, time, datetime, os, sys, pickle
from bert_train import bert_tokenize, create_data_loaders
from bert_class import BERTClass
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from util import plot_confusion_mat
import matplotlib.pyplot as plt


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(train_dataloader, validation_dataloader, model, label_embeddings, device, epochs, parent_child=None):
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed_all(seed_val)

    training_stats = []

    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):
        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...')

        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),
                      flush=True)

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            loss, logits = model(b_input_ids,
                                 label_embeddings,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels,
                                 device=device,
                                 parent_child=parent_child)

            total_train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epcoh took: {:}".format(training_time), flush=True)

        # ========================================
        #               Validation
        # ========================================
        print("", flush=True)
        print("Running Validation...", flush=True)

        t0 = time.time()
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                loss, logits = model(b_input_ids,
                                     label_embeddings,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels,
                                     device=device,
                                     parent_child=parent_child)

            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy), flush=True)

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
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("", flush=True)
    print("Training complete!", flush=True)

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)), flush=True)
    return model


def create_label_embeddings(glove_dir, index_to_label, device, label_word_map=None):
    embeddings = {}
    with open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')

            # Normalized glove embeddings
            embeddings[word] = coefs

    label_embeddings = []
    if label_word_map is None:
        for i in index_to_label:
            label_embeddings.append(embeddings[index_to_label[i]])
    else:
        for i in index_to_label:
            words = label_word_map[index_to_label[i]].split(",")
            temp_vec = 0
            for t in words:
                temp_vec += embeddings[t.strip()]
            label_embeddings.append(temp_vec / len(words))

    label_embeddings = torch.tensor(label_embeddings).to(device)
    return label_embeddings


def evaluate(model, prediction_dataloader, label_embeddings, device, parent_child=None):
    # Prediction on test set
    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            loss, logits = model(b_input_ids,
                                 label_embeddings,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=None,
                                 device=device,
                                 parent_child=parent_child)

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    return predictions, true_labels


def test(df_test, tokenizer, model, label_embeddings, device, label_to_index, index_to_label, parent_child=None):
    input_ids, attention_masks, labels = bert_tokenize(tokenizer, df_test, label_to_index)
    # Set the batch size.
    batch_size = 16
    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    predictions, true_labels = evaluate(model, prediction_dataloader, label_embeddings, device, parent_child)
    preds = []
    for pred in predictions:
        preds = preds + list(pred.argmax(axis=-1))
    true = []
    for t in true_labels:
        true = true + list(t)

    assert len(preds) == len(true)

    for i, t in enumerate(true):
        true[i] = index_to_label[t]
        preds[i] = index_to_label[preds[i]]

    return true, preds


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
    df_train, df_test = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)

    # Tokenize all of the sentences and map the tokens to their word IDs.
    print('Loading BERT tokenizer...', flush=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    label_set = set(df_train.label.values)
    label_to_index = {}
    index_to_label = {}
    for i, l in enumerate(list(label_set)):
        label_to_index[l] = i
        index_to_label[i] = l

    # label_word_map = json.load(open(pkl_dump_dir + "label_word_map.json", "r"))
    label_embeddings = create_label_embeddings(glove_dir, index_to_label, device)

    input_ids, attention_masks, labels = bert_tokenize(tokenizer, df_train, label_to_index)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-validation split.
    train_dataloader, validation_dataloader = create_data_loaders(dataset)

    model = BERTClass()
    model.to(device)

    model = train(train_dataloader, validation_dataloader, model, label_embeddings, device, epochs=5)

    true, preds = test(df_test, tokenizer, model, label_embeddings, device, label_to_index, index_to_label)
    print(classification_report(true, preds), flush=True)

    plot_confusion_mat(df_test["label"], preds, list(label_set))
    plt.savefig("./conf_mat.png")

    true, preds = test(df_train, tokenizer, model, label_embeddings, device, label_to_index, index_to_label)
    print(classification_report(true, preds), flush=True)

    tokenizer.save_pretrained(tok_path)
    torch.save(model, model_path + model_name)
