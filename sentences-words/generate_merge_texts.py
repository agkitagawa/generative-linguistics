from nltk.tree import Tree
import os
from pathlib import Path
from transformers import RobertaModel, RobertaTokenizer

model_name = "roberta-base" 
tokenizer = RobertaTokenizer.from_pretrained(model_name)

folder_path = Path('../PTB/WSJ')
output_folder = "../PTB/WSJ_sentence_samples"
suff_size_text = []
suff_size_trees = []
no_space_before = ",.:;?!)]}'/-"
no_space_after = "$@%#_([{'/-"

for root, dirs, files in os.walk(folder_path):
    for filename in files:
        file_path = os.path.join(root, filename)
        with open(file_path, 'r', encoding='ISO-8859-1') as infile:
            lines = infile.readlines() 

        trees = []
        curr_line = ""

        stack = []
        for line in lines:
            for character in line:
                curr_line += character
                if character == "(":
                    stack.append(character)
                if character == ")":
                    stack.pop()
                    if not stack:
                        trees.append(curr_line)
                        curr_line = ""
        trees.append(curr_line)

        trees = trees[1:]
        text_body = ""

        for index, tree in enumerate(trees):
            if tree.strip():
                curr = Tree.fromstring(tree) 
                trees[index] = curr
                leaves = curr.leaves()
                sentence = ""
                for element in leaves:
                    if element in no_space_before:
                        sentence = sentence[:-1]
                    elif element[0] == "*":
                        continue
                    elif element in no_space_after:
                        sentence += next_el

                    next_el = element + " "
                    sentence += next_el
           
                if sentence:
                    suff_size_text.append(sentence)
                    suff_size_trees.append(tree)

count = 0
for index, trees in enumerate(suff_size_trees):
    with open(f"{output_folder}/merge_trees/sentence_merge_tree_{count}.txt", "w") as file:
        for tree in trees:
            file.write(str(tree))
    with open(f"{output_folder}/texts/sentence_text_{count}.txt", "w") as file:
        text_body = suff_size_text[index]
        for sentence in text_body:
            file.write(sentence)
    count += 1
