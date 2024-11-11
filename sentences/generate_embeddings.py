import os
from pathlib import Path
from transformers import RobertaModel, RobertaTokenizer
import torch
import numpy as np

model_name = "roberta-base" 
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

output_folder = "/Users/annakitagawa/Downloads/Research/PTB/WSJ_sentence_samples"
text_folder = Path(f'{output_folder}/texts')
embedding_folder = f"{output_folder}/embeddings"

count = 0
for root, dirs, files in os.walk(text_folder):
    for filename in files:
        file_path = os.path.join(root, filename)
        with open(file_path, 'r') as infile:
            lines = infile.readlines() 
        
        inputs = []
        curr_line = ""

        for line in lines:
            input_ids = tokenizer.encode(line, add_special_tokens=True)
            input_ids_trimmed = input_ids[1:-1]

            input_ids_tensor = torch.tensor([input_ids_trimmed])
            attention_mask = torch.ones_like(input_ids_tensor)

            with torch.no_grad():
                outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask)

        embeddings = outputs.last_hidden_state
        sentence_embedding_np = embeddings.numpy()

        with open(f"{embedding_folder}/sentence_sample_embeddings_{count}.txt", "w") as file:
            np.savetxt(file, sentence_embedding_np[0], fmt="%.8f")
        
        count += 1
