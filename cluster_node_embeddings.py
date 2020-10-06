import json
from classifier_seedwords import preprocess_df
from get_skip_grams import encode_phrases, get_skipgrams, update_label_skipgram_clusters
from stellargraph.data import UniformRandomMetaPathWalk
from stellargraph.core.indexed_array import IndexedArray
from stellargraph import StellarGraph
from sklearn.mixture import GaussianMixture
from gensim.models import Word2Vec
from get_seed_words import decipher_phrase
import numpy as np
import pickle
import pandas as pd


def get_graph_metapaths(skip_gram_word_dict, skipgrams):
    skipgram_to_index = {}
    index_to_skipgram = {}
    sg_index = []
    for i, sg in enumerate(skipgrams):
        sg_index.append("sg" + str(i))
        skipgram_to_index[sg] = i
        index_to_skipgram[i] = sg
    sg_nodes = IndexedArray(np.array([[-1]] * len(sg_index)), index=sg_index)

    words = set()
    for sg in skipgrams:
        words.update(set(skip_gram_word_dict[sg]))

    word_to_index = {}
    index_to_word = {}
    word_index = []
    for i, word in enumerate(words):
        word_index.append("word" + str(i))
        word_to_index[word] = i
        index_to_word[i] = word
    word_nodes = IndexedArray(np.array([[-1]] * len(word_index)), index=word_index)

    source_nodes_list = []
    target_nodes_list = []
    for sg in skipgrams:
        for word in skip_gram_word_dict[sg]:
            source_nodes_list.append("sg" + str(skipgram_to_index[sg]))
            target_nodes_list.append("word" + str(word_to_index[word]))
    edges = pd.DataFrame({
        "source": source_nodes_list,
        "target": target_nodes_list
    })

    graph = StellarGraph({"sg": sg_nodes, "word": word_nodes}, edges)
    metapaths = [
        ["sg", "word", "sg"]
    ]
    return graph, metapaths, index_to_skipgram, index_to_word


if __name__ == "__main__":
    # base_path = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    base_path = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt"
    data_path = base_path + dataset + "/"

    df = pickle.load(open(data_path + "df_coarse_phrase.pkl", "rb"))
    df = preprocess_df(df)
    print("Loaded Dataframe..", flush=True)
    # tokenizer = fit_get_tokenizer(df.text, max_words=150000)
    label_words_phrases = json.load(open(data_path + "conwea_top100words_phrases.json", "r"))
    phrase_id = pickle.load(open(data_path + "phrase_id_coarse_map.pkl", "rb"))
    id_phrase_map = pickle.load(open(data_path + "id_phrase_coarse_map.pkl", "rb"))

    label_word_skip_gram_dict = pickle.load(open(data_path + "label_word_skip_gram_dict.pkl", "rb"))
    label_skip_gram_word_dict = pickle.load(open(data_path + "label_skip_gram_word_dict.pkl", "rb"))

    labels = list(set(df.label))
    print("Loaded skipgram maps..", flush=True)

    label_skipgram_clusters = {}
    for label in labels:
        print("Getting skipgrams for ", label, flush=True)
        label_skipgram_clusters[label] = {}
        word_skip_gram_dict = label_word_skip_gram_dict[label]
        skip_gram_word_dict = label_skip_gram_word_dict[label]
        entities = list(label_words_phrases[label].keys())
        encoded_entities = encode_phrases(entities, phrase_id)
        entity_skipgrams = get_skipgrams(encoded_entities, word_skip_gram_dict, skip_gram_word_dict, min_thresh=3,
                                         max_thresh=50)
        skipgram_entities = {}
        for e in entity_skipgrams:
            for sg in entity_skipgrams[e]:
                try:
                    skipgram_entities[sg].append(e)
                except:
                    skipgram_entities[sg] = [e]

        label_skipgrams = list(skipgram_entities.keys())

        graph, metapaths, index_to_skipgram, index_to_word = get_graph_metapaths(skip_gram_word_dict, label_skipgrams)
        print(
            "Number of nodes {} and number of edges {} in graph.".format(
                graph.number_of_nodes(), graph.number_of_edges()
            )
        )
        rw = UniformRandomMetaPathWalk(graph)
        walks = rw.run(
            nodes=list(graph.nodes()),  # root nodes
            length=5,  # maximum length of a random walk
            n=5,  # number of random walks per root node
            metapaths=metapaths,  # the metapaths
        )
        print("Number of random walks: {}".format(len(walks)))

        model = Word2Vec(walks, size=128, window=5, min_count=0, sg=1, workers=2, iter=10)
        print("Embeddings shape: ", model.wv.vectors.shape)
        node_ids = model.wv.index2word  # list of node IDs
        node_embeddings = (
            model.wv.vectors
        )

        sg_node_embeddings = []
        for i in index_to_skipgram:
            sg_node_embeddings.append(model["sg" + str(i)])

        if len(sg_node_embeddings) < 10:
            n_comps = len(sg_node_embeddings)
        else:
            n_comps = 10
        clf = GaussianMixture(n_components=n_comps, covariance_type="tied", init_params='kmeans', max_iter=100000)
        print("Clustering skipgrams..", flush=True)
        clf.fit(sg_node_embeddings)
        idx = clf.predict(sg_node_embeddings)
        label_skipgram_clusters = update_label_skipgram_clusters(label_skipgram_clusters, label_skipgrams, idx,
                                                                 skipgram_entities, id_phrase_map)

    pickle.dump(label_skipgram_clusters, open(data_path + "label_skipgram_clusters.pkl", "wb"))
    json.dump(label_skipgram_clusters, open(data_path + "label_skipgram_clusters.json", "w"))
