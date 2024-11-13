import os
from pathlib import Path
from transformers import RobertaModel, RobertaTokenizer
import torch
import numpy as np

model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

input_folder = "/Users/annakitagawa/Downloads/Research/PTB/WSJ_sentence_samples"
output_folder = "/Users/annakitagawa/Downloads/Research/PTB/WSJ_sentence_words_samples"
text_folder = Path(f'{input_folder}/texts')
embedding_folder = Path(f"{output_folder}/embeddings")

embedding_folder.mkdir(parents=True, exist_ok=True)  # Ensure the embedding folder exists

for root, dirs, files in os.walk(text_folder):
    for filename in files:
        file_path = os.path.join(root, filename)
        with open(file_path, 'r') as infile:
            lines = infile.readlines()

        for line in lines:

            # Tokenize the whole sentence
            word_tokens = tokenizer.tokenize(line.strip())  # Use strip() to remove any extra spaces or newlines

            if not word_tokens:  # If tokenization fails, skip the line
                print("No tokens found, skipping line.")
                continue

            word_embeddings = []
            word_start = 0
            words = line.split()

            current_word_tokens = []  # Holds tokens for the current word

            for i, token in enumerate(word_tokens):
                if token.startswith('Ġ'):
                    # If we find a token starting with 'Ġ', it marks a new word
                    # Process the previous word if there were tokens accumulated
                    if current_word_tokens:
                        word_token_indices = list(range(word_start, word_start + len(current_word_tokens)))
                        word_start = i
                        
                        # Get embeddings for accumulated tokens
                        input_ids = tokenizer.encode(line.strip(), add_special_tokens=True)
                        input_ids_tensor = torch.tensor([input_ids])
                        attention_mask = torch.ones_like(input_ids_tensor)

                        with torch.no_grad():
                            outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask)

                        embeddings = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]

                        word_embedding = embeddings[0, word_token_indices, :].mean(dim=0).numpy()
                        
                        if np.isnan(word_embedding).any():
                            print(f"NaN detected in embedding for word: {current_word_tokens}")
                            continue

                        word_embeddings.append(word_embedding)
                        current_word_tokens = [token]  # Start a new word with the current token
                    else:
                        current_word_tokens.append(token)  # Continue accumulating tokens for the current word
                else:
                    current_word_tokens.append(token)  # Continue accumulating tokens for the current word

            # Don't forget to process the last word
            if current_word_tokens:
                word_token_indices = list(range(word_start, word_start + len(current_word_tokens)))
                input_ids = tokenizer.encode(line.strip(), add_special_tokens=True)
                input_ids_tensor = torch.tensor([input_ids])
                attention_mask = torch.ones_like(input_ids_tensor)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask)

                embeddings = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]

                word_embedding = embeddings[0, word_token_indices, :].mean(dim=0).numpy()

                if np.isnan(word_embedding).any():
                    print(f"NaN detected in embedding for word: {current_word_tokens}")
                    continue

                word_embeddings.append(word_embedding)


            number = filename.split('_')[-1].split('.')[0]
            # Ensure that word_embeddings is not empty
            if word_embeddings:
                sentence_embedding_np = np.vstack(word_embeddings)
                with open(f"{embedding_folder}/sentence_words_embeddings_{number}.txt", "w") as file:
                    np.savetxt(file, sentence_embedding_np, fmt="%.8f")

