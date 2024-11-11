from nltk.tree import Tree

# need to make mst from merge trees - how?

sample_num = 1
file_path = f"/Users/annakitagawa/Downloads/Research/PTB/WSJ_medium_samples/merge_trees/medium_sample_merge_tree_{sample_num}.txt"

with open(file_path, 'r') as f:
    tree = Tree.fromstring(f.read())

tree.draw()