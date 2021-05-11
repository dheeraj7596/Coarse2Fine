from gpt2_ce import *

if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    basepath = "/data/dheeraj/coarse2fine/"
    dataset = sys.argv[4] + "/"
    pkl_dump_dir = basepath + dataset

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    algo = sys.argv[5]
    parent_label = sys.argv[6]
    tok_path = pkl_dump_dir + "gpt2/coarse_fine/" + algo + "/tokenizer"
    model_path = pkl_dump_dir + "gpt2/coarse_fine/" + algo + "/model/"
    model_name = "coarse_fine.pt"

    os.makedirs(tok_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    use_gpu = int(sys.argv[1])
    # use_gpu = False
    gpu_id = int(sys.argv[2])
    iteration = int(sys.argv[3])
    # iteration = 1

    # Tell pytorch to run this model on the GPU.
    if use_gpu:
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device("cpu")

    df = pickle.load(open(pkl_dump_dir + "df_coarse.pkl", "rb"))
    df = df[df.label.isin([parent_label])].reset_index(drop=True)
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', pad_token='<|pad|>',
                                              additional_special_tokens=['<|labelsep|>', '<|labelpad|>'])

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    parent_labels = [parent_label]
    child_labels = parent_to_child[parent_label]

    all_labels = parent_labels + child_labels

    pad_token_dict = {}
    max_num = -float("inf")
    for l in all_labels:
        tokens = tokenizer.tokenize(" ".join(l.split("_")))
        max_num = max(max_num, len(tokens))

    doc_start_ind = 1 + max_num + 1  # this gives the token from which the document starts in the inputids, 1 for the starttoken, max_num for label info, 1 for label_sup

    for l in all_labels:
        tokens = tokenizer.tokenize(" ".join(l.split("_")))
        pad_token_dict[l] = max_num - len(tokens)
    # tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<|startoftext|> sports <|labelsep|> Hello, my dog is cute <|endoftext|>")))

    df_weaksup = None
    for p in [parent_label]:
        for ch in parent_to_child[p]:
            temp_df = pickle.load(
                open(pkl_dump_dir + "exclusive/" + algo + "/" + str(iteration) + "it/" + ch + ".pkl", "rb"))
            temp_df["label"] = [ch] * len(temp_df)
            if df_weaksup is None:
                df_weaksup = temp_df
            else:
                df_weaksup = pd.concat([df_weaksup, temp_df])

    df = pd.concat([df, df_weaksup])
    coarse_input_ids, coarse_attention_masks = basic_gpt2_tokenize(tokenizer, df.text.values, df.label.values,
                                                                   pad_token_dict)
    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(coarse_input_ids, coarse_attention_masks)

    # Create a 90-10 train-validation split.
    coarse_train_dataloader, coarse_validation_dataloader = create_data_loaders(dataset, batch_size=4)

    model = train(model,
                  tokenizer,
                  coarse_train_dataloader,
                  coarse_validation_dataloader,
                  doc_start_ind,
                  all_labels,
                  device,
                  pad_token_dict)
    test_generate(model, tokenizer, all_labels, pad_token_dict, device)

    tokenizer.save_pretrained(tok_path)
    torch.save(model, model_path + model_name)
    pickle.dump(pad_token_dict, open(pkl_dump_dir + "pad_token_dict.pkl", "wb"))
