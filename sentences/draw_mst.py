import networkx as nx
import matplotlib.pyplot as plt

sample_num = 5
file_path = f"/Users/annakitagawa/Downloads/Research/PTB/WSJ_sentence_samples/msts/medium_sample_mst_{sample_num}.edgelist"

G = nx.read_edgelist(file_path)

nx.draw(G, with_labels=True, font_size=5, node_color='lightblue', node_size=10, edge_color='gray')
plt.title(f"Medium Sample MST {sample_num}")
plt.show()