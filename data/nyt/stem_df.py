import pickle
from nltk.stem.snowball import SnowballStemmer
import re
import string
from nltk.tokenize import sent_tokenize
import pandas as pd


def preprocess(text):
    stemmer = SnowballStemmer("english")
    sentences = sent_tokenize(text)
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    new_sentences = []
    for sent in sentences:
        sent = sent.lower()
        sent = re.sub(r'\d+', '', sent)
        sent = sent.translate(translator)
        new_sentence = []
        for w in sent.strip().split():
            new_sentence.append(stemmer.stem(w))
        new_sentence = " ".join(new_sentence)
        new_sentences.append(new_sentence)
    return " . ".join(new_sentences)


if __name__ == "__main__":
    data_path = "./"
    df_coarse = pickle.load(open(data_path + "df_coarse.pkl", "rb"))

    preprocessed_texts = []
    for i, row in df_coarse.iterrows():
        temp = preprocess(row["text"])
        if len(temp) == 0:
            print("No string")
            continue
        preprocessed_texts.append(temp)

    df = pd.DataFrame.from_dict({"text": preprocessed_texts, "label": df_coarse["label"]})
    pickle.dump(df, open(data_path + "df_coarse_stem.pkl", "wb"))
    pass
