from gensim.models import word2vec
from classifier_seedwords import preprocess_df
import pickle
from util import cosine_similarity
import math


def get_pseudo_label_surface_name(child_label_str, texts, embeddings, probability, child_label, den_nodes):
    candidate_words = set()
    for sent in texts:
        tokens = set(sent.strip().split())
        if child_label_str in tokens:
            candidate_words.update(tokens)

    candidate_words = candidate_words - {child_label_str}

    scores = []
    for c in candidate_words:
        cos_sim = cosine_similarity(embeddings[c], embeddings[child_label_str])
        if probability[child_label][c] == -math.inf:
            scores.append(-math.inf)
        else:
            den_score = 0
            for l in den_nodes:
                if probability[l][c] == -math.inf:
                    den_score = 1
                    break
                else:
                    den_score += probability[l][c]


if __name__ == "__main__":
    base_path = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # base_path = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt"
    data_path = base_path + dataset + "/"

    embedding_model = word2vec.Word2Vec.load(data_path + "word2vec.model")
    print("Loaded word2vec model..", flush=True)

    probability = pickle.load(open(data_path + "label_pmi_map.pkl", "rb"))
    df = pickle.load(open(data_path + "df_coarse_stem_phrase.pkl", "rb"))
    df = preprocess_df(df)
