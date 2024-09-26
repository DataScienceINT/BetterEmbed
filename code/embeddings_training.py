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

##### DATASET PREP #####
class ContrastiveQADataset(Dataset):
    def __init__(self, dataset, negative_samples=10):
        """
        Prepare pairs of (question, positive_answer, negative_answer) for contrastive training.
        :param dataset: JSON-like dataset where each question has one or more answers.
        :param negative_samples: Number of negative samples per question.
        """
        self.questions = []
        self.positive_answers = []
        self.negative_answers = []
        all_answers = []

        # Collect all possible answers across the dataset
        for entry in dataset.values():
            all_answers.extend(entry["answers"])

        # Create positive and negative pairs
        for entry in dataset.values():
            question = entry["question"]
            correct_answers = entry["answers"]

            for pos_answer in correct_answers:
                # Add positive pairs
                self.questions.append(question)
                self.positive_answers.append(pos_answer)
                
                # Sample negative answers
                neg_samples = random.sample([ans for ans in all_answers if ans not in correct_answers], negative_samples)
                self.negative_answers.append(neg_samples)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        positive_answer = self.positive_answers[idx]
        negative_answers = self.negative_answers[idx]
        
        return question, positive_answer, negative_answers


##### MODEL DESIGN #####

class ContrastiveAutoencoderConfig(PretrainedConfig):
    # This is to make our model compatible with HF
    def __init__(self, input_dim=768, embedding_dim=64, **kwargs):  # Update input_dim to match your embedding model's output size
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
class ContrastiveAutoencoder(PreTrainedModel):
    config_class = ContrastiveAutoencoderConfig

    def __init__(self, config, embedding_model):
        super().__init__(config)
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, 128),  
            nn.ReLU(),
            nn.Linear(128, config.embedding_dim)
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
        text_embeddings = text_embeddings.to(self.device)
        return self.encoder(text_embeddings)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Positive distance (between question and correct answer)
        pos_distance = torch.nn.functional.pairwise_distance(anchor, positive, p=2)
        # Negative distance (between question and incorrect answer)
        neg_distance = torch.nn.functional.pairwise_distance(anchor, negative, p=2)
        # Triplet Loss: minimize positive and maximize negative distance
        losses = torch.relu(pos_distance - neg_distance + self.margin)
        return losses.mean()


##### MODEL TRAINING ####

def train_contrastive_model(model, dataset, batch_size=32, epochs=10):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for questions, pos_answers, neg_answers in progress_bar:
            # Generate embeddings for questions, positive and negative answers
            anchor = model.encode(questions)
            positive = model.encode(pos_answers)
            negative = model.encode(neg_answers)

            # Compute triplet loss
            loss = criterion(anchor, positive, negative)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update progress bar description with the current loss
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    end_time = time.time()

    # Calculate total time taken
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    return model

##### SAVE MODEL (HF FORMAT) #####

def save_model_huggingface(model, tokenizer, save_directory):
    model.save_pretrained(save_directory)  # Save model weights and config
    tokenizer.save_pretrained(save_directory)  # Save tokenizer


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


    
    args = parser.parse_args()

    # Load the dataset
    with open(args.dataset, 'r') as file:
        dataset = json.load(file)

    # Prepare the dataset for contrastive training
    contrastive_dataset = ContrastiveQADataset(dataset, negative_samples=1)
    
    # Load the embedding model
    embedding_model = SentenceTransformer(args.embedding_model)

    #  Initialize the contrastive autoencoder configuration
    config = ContrastiveAutoencoderConfig(input_dim=args.input_dim, embedding_dim=args.embedding_dim)

    # Initialize the contrastive autoencoder model
    model = ContrastiveAutoencoder(config=config,embedding_model=embedding_model)

    # Train the model
    trained_model = train_contrastive_model(
        model=model,
        dataset=contrastive_dataset,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    # Save the trained model (HuggingFace format)
    tokenizer = embedding_model.tokenizer  # Get the tokenizer from the user-provided embedding model
    save_model_huggingface(trained_model, tokenizer, args.save_model)
    print(f"Model saved to {args.save_model}")

if __name__ == "__main__":
    main()

# Example running from the command line:
# python train_contrastive_model.py --dataset path_to_dataset.json --embedding_model paraphrase-MiniLM-L6-v2 --save_model trained_model.pth --batch_size 16 --epochs 5
