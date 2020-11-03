from gensim.models import word2vec
from classifier_seedwords import preprocess_df
import pickle
from util import cosine_similarity
from count_based_conditional_prob_stem import encode_into_phrase, decipher_phrase
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import math


def get_pseudo_label_surface_name(child_label_str, texts, embeddings, probability, parent, parent_labels, thresh=0.8):
    candidate_words = set()
    for sent in texts:
        tokens = set(sent.strip().split())
        if child_label_str in tokens:
            candidate_words.update(tokens)

    candidate_words = candidate_words - {child_label_str}
    filter_words = set([])
    for w in candidate_words:
        try:
            if cosine_similarity(embeddings[w], embeddings[child_label_str]) < thresh:
                filter_words.add(w)
        except Exception as e:
            print(e)
    candidate_words = candidate_words - filter_words

    candidate_words = list(candidate_words)
    scores = []

    try:
        child_label_thresh = probability[parent][child_label_str]
        den = 0
        for l in parent_labels:
            if l == parent:
                continue
            if child_label_str in probability[l] and probability[l][child_label_str] != -math.inf:
                den += probability[l][child_label_str]

        if den != 0:
            child_label_thresh = child_label_thresh / den
    except Exception as e:
        print(decipher_phrase(child_label_str, id_phrase_map), e)
        child_label_thresh = 0
    # child_label_thresh = 0

    for c in candidate_words:
        cos_sim = cosine_similarity(embeddings[c], embeddings[child_label_str])
        num = probability[parent][c]
        den = 0
        for l in parent_labels:
            if l == parent:
                continue
            if c in probability[l] and probability[l][c] != -math.inf:
                den += probability[l][c]
        if den != 0:
            val = cos_sim * (num / den)
        else:
            val = cos_sim * num

        if val > child_label_thresh:
            scores.append(val)

    inds = sorted(range(len(scores)), key=lambda i: scores[i])[-10:]
    words = []
    for i in inds:
        words.append(candidate_words[i])
    return words


if __name__ == "__main__":
    base_path = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # base_path = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt"
    data_path = base_path + dataset + "/"
    # Train a word2vec model
    embedding_model = word2vec.Word2Vec.load(data_path + "word2vec.model")
    print("Loaded word2vec model..", flush=True)

    df = pickle.load(open(data_path + "df_coarse_phrase_stem.pkl", "rb"))
    df = preprocess_df(df)
    probability = pickle.load(open(data_path + "label_pmi_map.pkl", "rb"))

    phrase_id = pickle.load(open(data_path + "phrase_id_coarse_map.pkl", "rb"))
    id_phrase_map = pickle.load(open(data_path + "id_phrase_coarse_map.pkl", "rb"))
    parent_to_child = pickle.load(open(data_path + "parent_to_child.pkl", "rb"))
    parent_labels = ["sports", "arts", "science", "business", "politics"]

    stemmer = SnowballStemmer("english")
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')

    for p in parent_labels:
        temp_df = df[df.label.isin([p])].reset_index(drop=True)
        for ch in parent_to_child[p]:
            child_label_str = encode_into_phrase(ch, phrase_id)
            print("For child label", ch)
            temp = get_pseudo_label_surface_name(child_label_str, temp_df.text, embedding_model, probability, p,
                                                 parent_labels)
            for i in temp:
                print(decipher_phrase(i, id_phrase_map))
            print("*" * 40)
