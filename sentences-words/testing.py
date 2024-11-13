import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


input_folder = "/Users/annakitagawa/Downloads/Research/PTB/WSJ_sentence_samples"
output_folder = "/Users/annakitagawa/Downloads/Research/PTB/WSJ_sentence_words_samples"
text_folder = f'{input_folder}/texts'
embeddings_folder = f"{output_folder}/embeddings"

mst_folder = f"{output_folder}/msts"

embedding_map = {}
word_map = {}  # To store word to embedding index mapping
count = 0

file_path = "/Users/annakitagawa/Downloads/Research/PTB/WSJ_sentence_words_samples/embeddings/sentence_words_embeddings_7.txt"
word_file_path = "/Users/annakitagawa/Downloads/Research/PTB/WSJ_sentence_samples/texts/sentence_text_7.txt"
with open(word_file_path, 'r') as word_file:
    words = word_file.read().split()  # Split by spaces to get the words

# Load the embeddings from the current embedding file
with open(file_path, 'r') as file:
    arr = np.loadtxt(file, dtype=np.float32)

print(arr)
print(words)
print(len(arr))
print(len(words))

# Map each embedding index to the corresponding word
for i, embedding in enumerate(arr):
    embedding_map[i] = embedding
    word_map[i] = words[i]  # Map the index to the corresponding word

num_embeddings = len(embedding_map)
distance_matrix = np.zeros((num_embeddings, num_embeddings))

# Calculate pairwise distances between embeddings
for i in range(num_embeddings):
    embedding1 = embedding_map[i]
    for j in range(num_embeddings):
        embedding2 = embedding_map[j]
        dist = np.linalg.norm(embedding1 - embedding2)
        distance_matrix[i, j] = dist

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        rootU = self.find(u)
        rootV = self.find(v)

        if rootU != rootV:
            if self.rank[rootU] > self.rank[rootV]:
                self.parent[rootV] = rootU
            elif self.rank[rootU] < self.rank[rootV]:
                self.parent[rootU] = rootV
            else:
                self.parent[rootV] = rootU
                self.rank[rootU] += 1

# Kruskal's algorithm to generate MST
def kruskal(num_points, distance_matrix):
    edges = []
    
    for i in range(num_points):
        for j in range(i + 1, num_points):
            edges.append((i, j, distance_matrix[i][j]))
    
    edges.sort(key=lambda x: x[2])
    
    uf = UnionFind(num_points)
    mst_edges = []
    
    for u, v, weight in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst_edges.append((u, v, weight)) 
    
    return mst_edges

mst = kruskal(num_embeddings, distance_matrix)

# Create neighbors dictionary and stripped edges list
stripped_edges = []
for v1, v2, dist in mst:
    stripped_edges.append((v1, v2))

# Create a graph using NetworkX
G = nx.Graph()
G.add_nodes_from(list(range(num_embeddings)))
G.add_edges_from(stripped_edges)

# Label the vertices with words
labels = {i: word_map[i] for i in range(num_embeddings)}

# Visualization of the graph
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, seed=42)  # Positioning for the nodes
nx.draw_networkx_nodes(G, pos, node_size=500)
nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color='b')
nx.draw_networkx_labels(G, pos, labels, font_size=12, font_color='black')

# Save the graph as an image
plt.axis('off')  # Turn off the axis
plt.savefig(f"{mst_folder}mst_{count}.png", format="PNG")

# Save the MST in edgelist format
file = f"{mst_folder}sentence_words_mst_{count}.edgelist"
nx.write_edgelist(G, file)

count += 1
