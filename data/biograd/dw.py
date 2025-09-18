import pandas as pd
import networkx as nx
import random
from gensim.models import Word2Vec

# -------- Step 1: Load Graph from Edge List --------
def load_graph(edge_list_path):
    G = nx.read_edgelist(edge_list_path)
    return G

# -------- Step 2: Generate Random Walks --------
def generate_random_walks(G, num_walks, walk_len):
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = random_walk(G, node, walk_len)
            walks.append(walk)
    return walks

def random_walk(G, start_node, walk_len):
    walk = [start_node]
    while len(walk) < walk_len:
        cur = walk[-1]
        neighbors = list(G.neighbors(cur))
        if neighbors:
            next_node = random.choice(neighbors)
            walk.append(next_node)
        else:
            break
    return walk

# -------- Step 3: Train Word2Vec (DeepWalk) --------
def train_deepwalk(walks, emb_size, window, workers, epochs):
    model = Word2Vec(
        sentences=walks,
        vector_size=emb_size,
        window=window,
        min_count=0,
        sg=1,  # Skip-gram
        workers=workers,
        epochs=epochs
    )
    return model

# -------- Step 4: Save Embeddings --------
def save_embeddings(model, output_path):
    model.wv.save_word2vec_format(output_path)

# -------- Parameters --------
walk_length = 40
num_walks_per_node = 40
embedding_size = 128
window_size = 5
workers = 4
epochs = 5

def deepwalk_pipeline(edge_list_path, output_path):
    G = load_graph(edge_list_path)
    walks = generate_random_walks(G, num_walks_per_node, walk_length)
    walks = [[str(node) for node in walk] for walk in walks]  # ensure string type
    model = train_deepwalk(walks, embedding_size, window_size, workers, epochs)
    save_embeddings(model, output_path)
    print(f"Embeddings saved to: {output_path}")

edge_path = '/itf-fi-ml/shared/users/ziyuzh/svm/data/biograd/biograd_entrz_2019.txt'
num_walks_per_node = 40
out_path = '/itf-fi-ml/shared/users/ziyuzh/svm/data/biograd/biograd_entrz_2019_dw_emb_'+str(num_walks_per_node)+'.txt'
deepwalk_pipeline(edge_path, out_path)