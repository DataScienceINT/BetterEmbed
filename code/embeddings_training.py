import random
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedModel, PretrainedConfig
import torch.optim as optim
import argparse
import json
from tqdm import tqdm
import time
import os
import faiss
import numpy as np

##### DATASET PREP #####

class ContrastiveQADataset(Dataset):
    def __init__(self, dataset):
        """
        Prepare pairs of (question, positive_answer) for contrastive training.
        :param dataset: JSON-like dataset where each question has one or more answers.
        """
        self.data = []  # This will store tuples of (question, positive_answer)

        # Prepare question and corresponding positive answer pairs
        for entry in dataset.values():  # Assuming each entry contains "question" and "answers"
            question = entry["question"]
            correct_answers = entry["answers"]

            # For each question, pair with each correct answer
            for pos_answer in correct_answers:
                self.data.append((question, pos_answer))  # Store as tuples in a list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return the question and its corresponding positive answer.
        """
        if isinstance(idx, int):
            return self.data[idx]
        elif isinstance(idx, list) or isinstance(idx, tuple):
            # If a batch of indices (list or tuple) is passed, handle each index
            return [self.data[i] for i in idx]
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")


##### FAISS INDEXING #####

def build_faiss_index(model, all_answers, use_gpu):
    """
    Build a FAISS index for all answers in the dataset. Use FAISS on GPU if available.
    :param model: The trained model to encode answers.
    :param all_answers: List of all possible answers.
    :param use_gpu: Whether to use FAISS on GPU (True) or CPU (False).
    :return: FAISS index and encoded answer embeddings.
    """
    # Encode all answers using the model
    answer_embeddings = model.encode(all_answers).detach().cpu().numpy()

    # Create FAISS index for cosine similarity (inner product)
    index = faiss.IndexFlatIP(answer_embeddings.shape[1])

    if use_gpu:
        # Move FAISS index to GPU if GPU is available
        res = faiss.StandardGpuResources()  # Initialize GPU resources
        index = faiss.index_cpu_to_gpu(res, 0, index)  # Transfer index to GPU

    # Add the answer embeddings to the FAISS index
    index.add(answer_embeddings)

    return index, answer_embeddings

def batched_faiss_search(index, embeddings, batch_size=512):
    """
    Perform batched FAISS search to avoid memory overload.
    :param index: FAISS index containing all answer embeddings.
    :param embeddings: Encoded embeddings to search for.
    :param batch_size: Size of each batch for FAISS search.
    :return: Nearest neighbors indices for each batch.
    """
    num_batches = len(embeddings) // batch_size + (1 if len(embeddings) % batch_size != 0 else 0)
    all_indices = []
    
    for i in range(num_batches):
        batch_embeddings = embeddings[i * batch_size: (i + 1) * batch_size]
        _, batch_indices = index.search(batch_embeddings, k=1)  # Adjust 'k' as needed for nearest neighbors
        all_indices.append(batch_indices)

    return np.vstack(all_indices)

def find_hard_negatives(index, model, questions, positives, all_answers, k=1, use_gpu=True, batch_size=512):
    # Modify find_hard_negatives to use batched FAISS search
    question_embeddings = model.encode(questions).detach().cpu().numpy()
    
    # Use batched FAISS search for efficiency
    nearest_indices = batched_faiss_search(index, question_embeddings, batch_size=batch_size)

    hard_negatives = []
    for i, indices in enumerate(nearest_indices):
        hard_negative_indices = [idx for idx in indices if all_answers[idx] != positives[i]]
        if len(hard_negative_indices) > 0:
            hard_negatives.append(all_answers[hard_negative_indices[0]])
        else:
            hard_negatives.append(random.choice(all_answers))  # Fallback

    return hard_negatives
    
##### MODEL DESIGN #####

class ContrastiveAutoencoderConfig(PretrainedConfig):
    # This is to make our model compatible with HF
    def __init__(self, input_dim=768, embedding_dim=64, **kwargs):  # input_dim to match your embedding model's output size
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
class ContrastiveAutoencoder(PreTrainedModel):
    config_class = ContrastiveAutoencoderConfig

    def __init__(self, config, embedding_model=None):
        super().__init__(config)
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, 128),  
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, config.embedding_dim),
            nn.BatchNorm1d(config.embedding_dim)
        )
        self.embedding_model = embedding_model

    def forward(self, x):
        return self.encoder(x)

    def encode(self, texts):
        """
        Generate embeddings directly from raw text using the user-provided embedding model + encoder.
        :param texts: List of text strings.
        :return: Encoded embeddings.
        """
        # Generate text embeddings using the SentenceTransformer
        text_embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)  
        text_embeddings = text_embeddings.to(next(self.parameters()).device)
        return self.encoder(text_embeddings)

    @classmethod
    def from_pretrained(cls, model_path, *model_args, **kwargs):
        """
        Load the ContrastiveAutoencoder along with the embedding model.
        """
        # Load the base ContrastiveAutoencoder model
        model = super().from_pretrained(model_path, *model_args, **kwargs)

        # Now manually load the SentenceTransformer embedding model from the 'embedding_model' subdirectory
        embedding_model_path = os.path.join(model_path, 'embedding_model')
        if os.path.exists(embedding_model_path):
            embedding_model = SentenceTransformer(embedding_model_path)
            model.embedding_model = embedding_model
        else:
            raise ValueError(f"Embedding model not found at {embedding_model_path}")

        return model


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Normalize embeddings to unit vectors
        anchor = nn.functional.normalize(anchor, p=2, dim=-1)
        positive = nn.functional.normalize(positive, p=2, dim=-1)
        negative = nn.functional.normalize(negative, p=2, dim=-1)
        
        # Positive similarity (between question and correct answer)
        pos_similarity = torch.nn.functional.cosine_similarity(anchor, positive, dim=-1)
        # Negative similarity (between question and incorrect answer)
        neg_similarity = torch.nn.functional.cosine_similarity(anchor, negative, dim=-1)
        # Triplet Loss: minimize negative and maximize positive similarity
        losses = torch.relu(neg_similarity - pos_similarity + self.margin)
        return losses.mean()


##### MODEL TRAINING ####

def train_contrastive_model(model, dataset, batch_size=32, epochs=10, faiss_batch_size=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = TripletLoss(margin=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    all_answers = list(set(answer for _, answer in dataset.data))
    faiss_index, all_answer_embeddings = build_faiss_index(model, all_answers, use_gpu=torch.cuda.is_available())

    start_time = time.time()

    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for questions, pos_answers in progress_bar:
            questions, pos_answers = list(questions), list(pos_answers)

            # Dynamically find hard negatives using batched FAISS search
            hard_negatives = find_hard_negatives(
                faiss_index, model, questions, pos_answers, all_answers, k=1, use_gpu=torch.cuda.is_available(), batch_size=faiss_batch_size
            )

            anchor = model.encode(questions).to(device)
            positive = model.encode(pos_answers).to(device)
            negative = model.encode(hard_negatives).to(device)

            loss = criterion(anchor, positive, negative)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    return model


##### SAVE MODEL (HF FORMAT) #####

def save_model_huggingface(model, save_directory):
    # Ensure the directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Save the ContrastiveAutoencoder model
    model.save_pretrained(save_directory)

    # Save the SentenceTransformer embedding model
    if model.embedding_model is not None:
        model.embedding_model.save(os.path.join(save_directory, 'embedding_model'))

        # Check if the embedding model has a tokenizer and save it if it does
        if hasattr(model.embedding_model, 'tokenizer'):
            model.embedding_model.tokenizer.save_pretrained(os.path.join(save_directory, 'embedding_model'))



##### PERFORM TRAINING #####

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train contrastive embeddings model for question-answer retrieval.")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset file (JSON format)")
    parser.add_argument('--embedding_model', type=str, required=True, help="Pre-trained embedding model ")
    parser.add_argument('--save_model', type=str, required=True, help="Path to save the trained model")
    parser.add_argument('--embedding_dim', type=int, default=128, help="Embedding dimension for the contrastive model (default: 128)")
    parser.add_argument('--input_dim', type=int, default=768, help="Input dimension for embeddings")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training (default: 10)")
    parser.add_argument('--faiss_batch_size', type=int, default=512, help="Batch size for FAISS search to avoid memory overload")

    args = parser.parse_args()

    # Load the dataset
    with open(args.dataset, 'r') as file:
        dataset = json.load(file)

    # Prepare the dataset for contrastive training
    contrastive_dataset = ContrastiveQADataset(dataset)

    # Load the embedding model
    embedding_model = SentenceTransformer(args.embedding_model)

    # Initialize the contrastive autoencoder configuration
    config = ContrastiveAutoencoderConfig(input_dim=args.input_dim, embedding_dim=args.embedding_dim)

    # Initialize the contrastive autoencoder model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveAutoencoder(config=config, embedding_model=embedding_model)
    model.to(device)

    # Train the model and pass faiss_batch_size as an argument
    trained_model = train_contrastive_model(
        model=model,
        dataset=contrastive_dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        faiss_batch_size=args.faiss_batch_size
    )

    # Save the trained model (HuggingFace format)
    save_model_huggingface(trained_model, args.save_model)
    print(f"Model saved to {args.save_model}")

if __name__ == "__main__":
    main()

# Example running from the command line:
# python embeddings_training.py  --dataset path_to_dataset.json --embedding_model NeuML/pubmedbert-base-embeddings --save_model path/to/model --faiss_batch_size 256

