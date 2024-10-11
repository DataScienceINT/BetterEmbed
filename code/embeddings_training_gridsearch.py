import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedModel, PretrainedConfig
from tqdm import tqdm
import numpy as np
import faiss
import argparse
import json
import os
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split


##### DATASET PREP #####
class ContrastiveQADataset(Dataset):
    """
    Prepares pairs of (question, positive_answer) for contrastive training from the dataset.
    """
    def __init__(self, dataset):
        self.data = []
        for entry in dataset.values():
            question = entry["question"]
            correct_answers = entry["answers"]
            for pos_answer in correct_answers:
                self.data.append((question, pos_answer))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


##### SEED SETTING #####
def set_seed(seed=42):
    """
    Set the seed for reproducibility across numpy, random, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


##### FAISS #####
def build_faiss_index(model, all_answers, use_gpu):
    """
    Builds a FAISS index for all answers using the same model that encodes questions, ensuring consistent dimensions.
    """
    # Encode answers using the same model to ensure consistent dimensionality with the questions
    answer_embeddings = model.encode(all_answers)

    # If the embeddings are PyTorch tensors, detach them and move to CPU if needed, then convert to NumPy
    if torch.is_tensor(answer_embeddings):
        answer_embeddings = answer_embeddings.detach().cpu().numpy()

    # Normalize embeddings for cosine similarity
    answer_embeddings = answer_embeddings / np.linalg.norm(answer_embeddings, axis=1, keepdims=True)

    # Inner product index for approximation of cosine similarity
    index = faiss.IndexFlatIP(answer_embeddings.shape[1])

    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(answer_embeddings) 
    return index, answer_embeddings


def batched_faiss_search(index, embeddings, batch_size=512):
    """
    Performs FAISS search in batches to avoid memory overload.
    Returns nearest neighbors indices for the embeddings.
    """
    num_batches = len(embeddings) // batch_size + (1 if len(embeddings) % batch_size != 0 else 0)
    all_indices = []
    for i in range(num_batches):
        batch_embeddings = embeddings[i * batch_size: (i + 1) * batch_size]
        _, batch_indices = index.search(batch_embeddings, k=3)
        all_indices.append(batch_indices)
    return np.vstack(all_indices)


def reset_faiss_index(index):
    """
    Resets FAISS index by deleting it if necessary.
    """
    if index is not None:
        del index
        
    # Clean up GPU memory if necessary
    torch.cuda.empty_cache()  
    
##### HARD NEGATIVE MINING #####
def find_hard_negatives(index, model, questions, positives, all_answers, batch_size=512, k=5):
    """
    Finds hard negatives (incorrect but similar answers) using FAISS for given questions.
    Normalizes the question embeddings for cosine similarity search.
    Arguments:
    - k: the number of nearest neighbors to consider for finding negatives
    """
    # Encode questions using the same model as answers
    question_embeddings = model.encode(questions)
    
    # If the embeddings are tensors, detach and move them to CPU, then convert to NumPy
    if torch.is_tensor(question_embeddings):
        question_embeddings = question_embeddings.detach().cpu().numpy()

    # Normalize the embeddings for cosine similarity
    question_embeddings = question_embeddings / np.linalg.norm(question_embeddings, axis=1, keepdims=True)

    # Perform FAISS search to get the top k nearest neighbors for each question
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
    """
    Configuration class for Contrastive Autoencoder, inheriting from Huggingface's PretrainedConfig.
    """
    def __init__(self, input_dim=768, embedding_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim


class ContrastiveAutoencoder(PreTrainedModel):
    """
    Contrastive Autoencoder model for learning embeddings.
    """
    config_class = ContrastiveAutoencoderConfig

    def __init__(self, config, embedding_model=None, dropout_p=0.3):
        super().__init__(config)
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
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


##### TRIPLET LOSS #####
class TripletLoss(nn.Module):
    """
    Triplet loss function that encourages positive pairs to be closer than negative pairs.
    Assumes that anchor, positive, and negative embeddings are already normalized.
    """
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Assumes anchor, positive, and negative are already normalized
        pos_similarity = torch.nn.functional.cosine_similarity(anchor, positive, dim=-1)
        neg_similarity = torch.nn.functional.cosine_similarity(anchor, negative, dim=-1)
        losses = torch.relu(neg_similarity - pos_similarity + self.margin)
        return losses.mean()


##### TRAINING AND EVALUATION #####
def evaluate_model(model, val_loader, criterion, device, faiss_index, all_answers, faiss_batch_size):
    """
    Evaluates the model on the validation set and computes validation loss.
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for questions, pos_answers in val_loader:
            questions, pos_answers = list(questions), list(pos_answers)
            anchor = model.encode(questions).to(device)
            positive = model.encode(pos_answers).to(device)
            hard_negatives = find_hard_negatives(faiss_index, model, questions, pos_answers, all_answers, batch_size=faiss_batch_size)
            negative = model.encode(hard_negatives).to(device)
            loss = criterion(anchor, positive, negative)
            val_loss += loss.item()
    return val_loss / len(val_loader)


def train_contrastive_model(model, train_loader, val_loader, optimizer, criterion, scheduler, device, faiss_index, all_answers, faiss_batch_size, patience=5, epochs=10, log_file=None):
    """
    Trains the contrastive model with early stopping based on validation loss.
    Returns the trained model along with train and validation losses.
    Logs progress into the specified log file.
    """
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    
    for epoch in range(1, epochs + 10):
        model.train()
        total_loss = 0.0
        
        for questions, pos_answers in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            questions, pos_answers = list(questions), list(pos_answers)

            # Find hard negatives using FAISS
            hard_negatives = find_hard_negatives(faiss_index, model, questions, pos_answers, all_answers, batch_size=faiss_batch_size)

            # Encode anchor (questions), positive (correct answers), and negative (hard negatives)
            anchor = model.encode(questions).to(device)   
            positive = model.encode(pos_answers).to(device)
            negative = model.encode(hard_negatives).to(device)  

            # Calculate loss and optimize
            optimizer.zero_grad()
            loss = criterion(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Step
        avg_val_loss = evaluate_model(model, val_loader, criterion, device, faiss_index, all_answers, faiss_batch_size)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        # Log losses
        log_message = f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}\n"
        print(log_message)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(log_message)

        # Early stopping condition
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return model, train_losses, val_losses



##### SAVING BEST MODEL #####
def save_best_model(model, tokenizer, save_dir):
    """
    Saves the model and tokenizer in HuggingFace format. If a directory exists, it empties it first.
    """
    # Check if the directory exists; if so, empty it
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    # Save model and tokenizer in HuggingFace format
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Best model saved at {save_dir}")


##### PLOTTING LOSSES #####
def plot_losses(grid_results, save_path):
    """
    Plots training and validation losses for each hyperparameters combination as small multiples.
    """
    num_combinations = len(grid_results)
    cols = 3  # Number of columns for subplots
    rows = (num_combinations + cols - 1) // cols  # Calculate number of rows needed
    
    fig, axs = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axs = axs.flatten()

    for i, result in enumerate(grid_results):
        label = f"lr={result['params']['lr']}, margin={result['params']['margin']}, emb_dim={result['params']['embedding_dim']}"
        axs[i].plot(result['train_losses'], label="Train")
        axs[i].plot(result['val_losses'], label="Val", linestyle='dashed')
        axs[i].set_title(label)
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel('Loss')
        axs[i].legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)


##### SAVING BEST PARAMETERS #####
def save_best_params(best_params, save_path):
    """
    Saves best hyperparameters to file for future reference
    """
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)


##### MAIN FUNCTION #####
def main():
    parser = argparse.ArgumentParser(description="Train contrastive embeddings model with train-validation split.")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset file (JSON format)")
    parser.add_argument('--embedding_model', type=str, required=True, help="Pre-trained embedding model")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to store logs, model, and plots")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training (default: 10)")
    parser.add_argument('--lr', type=float, nargs='+', default=[1e-3, 1e-4], help="Learning rates to test")
    parser.add_argument('--margin', type=float, nargs='+', default=[0.2, 0.5], help="Margins to test")
    parser.add_argument('--embedding_dim', type=int, nargs='+', default=[64, 128], help="Embedding dimensions to test")
    parser.add_argument('--faiss_batch_size', type=int, default=512, help="Batch size for FAISS search")
    parser.add_argument('--patience', type=int, default=5, help="Patience for early stopping")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")

    args = parser.parse_args()
    set_seed(args.seed)

    # Load the dataset
    with open(args.dataset, 'r') as file:
        dataset = json.load(file)
    contrastive_dataset = ContrastiveQADataset(dataset)

    # Split dataset into train and validation (e.g. 80% train, 20% validation)
    train_size = int(0.8 * len(contrastive_dataset))
    val_size = len(contrastive_dataset) - train_size
    train_dataset, val_dataset = random_split(contrastive_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Load embedding model and tokenizer
    embedding_model = SentenceTransformer(args.embedding_model)
    tokenizer = embedding_model.tokenizer  # Assuming the tokenizer is available

    # Grid search setup
    grid_results = []
    best_params = None
    best_val_loss = float('inf')

    for lr in args.lr:
        for margin in args.margin:
            for embedding_dim in args.embedding_dim:
                param_dir = os.path.join(args.output_dir, f"lr_{lr}_margin_{margin}_embdim_{embedding_dim}")
                os.makedirs(param_dir, exist_ok=True)
                log_file = os.path.join(param_dir, "training_log.txt")

                config = ContrastiveAutoencoderConfig(input_dim=768, embedding_dim=embedding_dim)
                model = ContrastiveAutoencoder(config=config, embedding_model=embedding_model).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                optimizer = optim.Adam(model.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)
                criterion = TripletLoss(margin=margin).to(model.device)

                # Build FAISS index for the current hyperparameter set
                all_answers = list(set(answer for _, answer in contrastive_dataset.data))
                faiss_index, _ = build_faiss_index(model, all_answers, use_gpu=torch.cuda.is_available())

                model, train_losses, val_losses = train_contrastive_model(
                    model, train_loader, val_loader, optimizer, criterion, scheduler, model.device, faiss_index, all_answers, args.faiss_batch_size, patience=args.patience, epochs=args.epochs, log_file=log_file
                )

                avg_val_loss = val_losses[-1]
                grid_results.append({
                    'params': {'lr': lr, 'margin': margin, 'embedding_dim': embedding_dim},
                    'train_losses': train_losses,
                    'val_losses': val_losses
                })

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_params = {'lr': lr, 'margin': margin, 'embedding_dim': embedding_dim}
                    # Save the best model and tokenizer
                    save_best_model(model, tokenizer, os.path.join(param_dir, 'best_model'))

                save_best_params(best_params, os.path.join(param_dir, 'best_params.json'))
                plot_losses(grid_results, os.path.join(param_dir, 'losses_plot.png'))


if __name__ == "__main__":
    main()
