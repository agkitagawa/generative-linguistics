import networkx as nx
import matplotlib.pyplot as plt
from nltk.tree import Tree

sample_num = 581
file_path = f"/Users/annakitagawa/Downloads/Research/PTB/WSJ_sentence_words_samples/msts/sentence_words_mst_{sample_num}.edgelist"

G = nx.read_edgelist(file_path)

nx.draw(G, with_labels=True, font_size=5, node_color='lightblue', node_size=10, edge_color='gray')
plt.title(f"Medium Sample MST {sample_num}")
plt.show()


# file_path = f"/Users/annakitagawa/Downloads/Research/PTB/WSJ_sentence_samples/merge_trees/sentence_merge_tree_{sample_num}.txt"

# with open(file_path, 'r') as f:
#     tree = Tree.fromstring(f.read())

# tree.draw()