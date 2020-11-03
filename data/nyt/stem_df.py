import pickle
from nltk.stem.snowball import SnowballStemmer
import re
import string
from nltk.tokenize import sent_tokenize
import pandas as pd
from parse_autophrase_output import decrypt
from get_seed_words import decipher_phrase

def preprocess(text, id_phrase):
    stemmer = SnowballStemmer("english")
    sentences = sent_tokenize(text)
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    new_sentences = []
    for sent in sentences:
        sent = sent.lower()
        sent = re.sub(r'\b\d+\b', '', sent)
        sent = sent.translate(translator)
        new_sentence = []
        for w in sent.strip().split():
            if decipher_phrase(w, id_phrase) == w:
                new_sentence.append(stemmer.stem(w))
            else:
                new_sentence.append(w)
        new_sentence = " ".join(new_sentence)
        new_sentences.append(new_sentence)
    return " . ".join(new_sentences)


if __name__ == "__main__":
    data_path = "./"
    id_phrase = pickle.load(open(data_path + "id_phrase_coarse_map.pkl", "rb"))
    df_coarse = pickle.load(open(data_path + "df_coarse_phrase.pkl", "rb"))

    preprocessed_texts = []
    for i, row in df_coarse.iterrows():
        temp = preprocess(row["text"], id_phrase)
        if len(temp) == 0:
            print("No string")
            continue
        preprocessed_texts.append(temp)

    df = pd.DataFrame.from_dict({"text": preprocessed_texts, "label": df_coarse["label"]})
    pickle.dump(df, open(data_path + "df_coarse_phrase_stem.pkl", "wb"))
    pass
