import sys
import torch
import pickle
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup, AdamW
from sklearn.metrics import classification_report
from gpt2_coarse_finetune import gpt2_tokenize, test_generate, create_data_loaders
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch._utils import _accumulate
from torch._C._VariableFunctions import randperm
from torch.utils.data.dataset import Subset
from torch._C import default_generator
import numpy as np
import random
import time


def gpt2_fine_tokenize(tokenizer, df, label_to_index, index_to_label, max_length=768):
    input_ids = []
    attention_masks = []
    # For every sentence...
    sentences = df.text.values

    for i, sent in enumerate(sentences):
        sibling_input_ids = []
        sibling_attn_masks = []
        for ind_l in index_to_label:
            encoded_dict = tokenizer.encode_plus(
                " ".join(index_to_label[ind_l].split("_")) + " <|labelsep|> " + sent,  # Sentence to encode.
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
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


def train(coarse_model, fine_model, coarse_tokenizer, fine_tokenizer, train_dataloader, validation_dataloader, device):
    optimizer = AdamW(fine_model.parameters(),
                      lr=5e-4,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )

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

    fine_posterior = torch.ones()

    for epoch_i in range(0, epochs):
        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)
        t0 = time.time()
        total_train_loss = 0
        fine_model.train()

        for step, batch in enumerate(train_dataloader):
            # todo write what to do if step%sample_every case

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_input_mask = batch[1].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            loss = outputs[0]
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()


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
    coarse_model = torch.load(model_path + model_name)
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

        temp_df = df[df.label.isin(children)].reset_index(drop=True)
        coarse_input_ids, coarse_attention_masks = gpt2_tokenize(coarse_tokenizer, temp_df)
        fine_input_ids, fine_attention_masks = gpt2_fine_tokenize(fine_tokenizer, temp_df, label_to_index,
                                                                  index_to_label)
        dataset = TensorDataset(coarse_input_ids, coarse_attention_masks, fine_input_ids, fine_attention_masks)

        train_dataloader, validation_dataloader = create_data_loaders(dataset, batch_size=4)
        fine_posterior, fine_model = train(coarse_model,
                                           fine_model,
                                           coarse_tokenizer,
                                           fine_tokenizer,
                                           train_dataloader,
                                           validation_dataloader,
                                           device)
        test_generate(fine_model, fine_tokenizer, children, device)
        true, preds = test(fine_model, fine_tokenizer, fine_posterior, temp_df, device)
        all_true += true
        all_preds += preds

        # todo save these at right location
        fine_tokenizer.save_pretrained(tok_path)
        torch.save(fine_model, model_path + model_name)

    print(classification_report(all_true, all_preds), flush=True)
    print("*" * 80, flush=True)
