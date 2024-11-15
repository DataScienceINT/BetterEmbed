import random
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedModel, PretrainedConfig
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import faiss
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import argparse
from sklearn.model_selection import train_test_split


##### DATASET PREP #####

class ContrastiveQADataset(Dataset):
    def __init__(self, dataset, split='train'):
        """
        Prepare pairs of (question, positive_answer) for contrastive training.
        Also store all answers for this split to use in negative mining.
        """
        self.data = []
        self.all_answers = set()
        self.split = split
        
        for entry in dataset.values():
            question = entry["question"]
            correct_answers = entry["answers"]
            for pos_answer in correct_answers:
                self.data.append((question, pos_answer))
                self.all_answers.add(pos_answer)
        
        self.all_answers = list(self.all_answers)
        self.answer_to_idx = {ans: idx for idx, ans in enumerate(self.all_answers)}

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
    

##### FAISS INDEXING #####

def build_answer_faiss_index(model, answers, use_gpu):
    """
    Build a FAISS index for answer embeddings.
    """
    print(f"Building FAISS index for {len(answers)} answers...")
    answer_embeddings = model.encode(answers).detach().cpu().numpy()
    index = faiss.IndexFlatIP(answer_embeddings.shape[1])
    if use_gpu and torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(answer_embeddings)
    return index, answer_embeddings

def batched_faiss_search(index, embeddings, batch_size=512, k=3):
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
        _, batch_indices = index.search(batch_embeddings, k=3)  # Adjust 'k' as needed for nearest neighbors
        all_indices.append(batch_indices)

    return np.vstack(all_indices)

def find_hard_negatives(index, model, questions, positives, all_answers, k=3, use_gpu=True, batch_size=512):
    # Modify find_hard_negatives to use batched FAISS search
    question_embeddings = model.encode(questions).detach().cpu().numpy()
    
    # Use batched FAISS search for efficiency
    nearest_indices = batched_faiss_search(index, question_embeddings, batch_size=batch_size, k=k)

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
    def __init__(self, input_dim=768, embedding_dim=128, **kwargs):  # input_dim to match your embedding model's output size
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
        
    def encode(self, texts):
        """Encode texts using the full pipeline."""
        with torch.set_grad_enabled(self.training):
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
            embeddings = embeddings.to(self.device)
            return self.encoder(embeddings)
    
    def forward(self, texts):
        """Forward pass for the model."""
        return self.encode(texts)
    
    def save_pretrained(self, save_directory):
        """Save model in HuggingFace format."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the config
        self.save_pretrained(save_directory)
        
        # Save the SentenceTransformer embedding model
        if model.embedding_model is not None:
            model.embedding_model.save(os.path.join(save_directory, 'embedding_model'))
    
            # Check if the embedding model has a tokenizer and save it if it does
            if hasattr(model.embedding_model, 'tokenizer'):
                model.embedding_model.tokenizer.save_pretrained(os.path.join(save_directory, 'embedding_model'))

    
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
        anchor = nn.functional.normalize(anchor, p=2, dim=-1)
        positive = nn.functional.normalize(positive, p=2, dim=-1)
        negative = nn.functional.normalize(negative, p=2, dim=-1)
        pos_similarity = torch.nn.functional.cosine_similarity(anchor, positive, dim=-1)
        neg_similarity = torch.nn.functional.cosine_similarity(anchor, negative, dim=-1)
        losses = torch.relu(neg_similarity - pos_similarity + self.margin)
        return losses.mean()

##### MODEL TRAINING ####

class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamped log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.loss_log_path = os.path.join(log_dir, f'losses_{timestamp}.json')
        self.metrics_log_path = os.path.join(log_dir, f'metrics_{timestamp}.json')
        self.plot_path = os.path.join(log_dir, f'training_plot_{timestamp}.png')
        
        self.logs = {
            'losses': {'train': [], 'val': []},
            'metrics': {'train': [], 'val': []},
            'config': {}
        }

    def log_losses(self, train_loss, val_loss, epoch):
        self.logs['losses']['train'].append({'epoch': epoch, 'loss': train_loss})
        self.logs['losses']['val'].append({'epoch': epoch, 'loss': val_loss})
        self._save_logs()

    def log_metrics(self, metrics, split, epoch):
        self.logs['metrics'][split].append({'epoch': epoch, **metrics})
        self._save_logs()

    def log_config(self, config):
        self.logs['config'] = config
        self._save_logs()

    def _save_logs(self):
        # Save losses
        with open(self.loss_log_path, 'w') as f:
            json.dump(self.logs['losses'], f, indent=2)
        
        # Save metrics
        with open(self.metrics_log_path, 'w') as f:
            json.dump(self.logs['metrics'], f, indent=2)

    def plot_training_progress(self):
        epochs = range(1, len(self.logs['losses']['train']) + 1)
        train_losses = [log['loss'] for log in self.logs['losses']['train']]
        val_losses = [log['loss'] for log in self.logs['losses']['val']]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.plot_path)
        plt.close()

def train_contrastive_model(
    model, 
    train_loader, 
    val_loader,
    optimizer, 
    criterion, 
    scheduler, 
    device, 
    train_dataset,
    val_dataset,
    logger,
    save_dir,
    patience=5, 
    epochs=10,
    faiss_batch_size=256
):
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Build FAISS indices
    train_faiss_index, _ = build_answer_faiss_index(
        model, 
        train_dataset.all_answers,
        use_gpu=torch.cuda.is_available()
    )
    
    val_faiss_index, _ = build_answer_faiss_index(
        model, 
        val_dataset.all_answers,
        use_gpu=torch.cuda.is_available()
    )

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        total_train_loss = 0.0
        
        for questions, pos_answers in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            questions, pos_answers = list(questions), list(pos_answers)
            
            hard_negatives = find_hard_negatives(
                train_faiss_index,
                model,
                pos_answers,
                train_dataset.all_answers,
                faiss_batch_size
            )
            
            anchor = model.encode(questions).to(device)
            positive = model.encode(pos_answers).to(device)
            negative = model.encode(hard_negatives).to(device)
            
            optimizer.zero_grad()
            loss = criterion(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for questions, pos_answers in val_loader:
                questions, pos_answers = list(questions), list(pos_answers)
                
                hard_negatives = find_hard_negatives(
                    val_faiss_index,
                    model,
                    pos_answers,
                    val_dataset.all_answers
                )
                
                anchor = model.encode(questions).to(device)
                positive = model.encode(pos_answers).to(device)
                negative = model.encode(hard_negatives).to(device)
                
                loss = criterion(anchor, positive, negative)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Log losses and update plot
        logger.log_losses(avg_train_loss, avg_val_loss, epoch)
        logger.plot_training_progress()
        
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model in HuggingFace format
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_dir = os.path.join(save_dir, 'best_model')
            model.save_pretrained(best_model_dir)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Save final model in HuggingFace format
    final_model_dir = os.path.join(save_dir, 'final_model')
    model.save_pretrained(final_model_dir)
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train contrastive embeddings model")
    parser.add_argument('--dataset', type=str, required=True, help="Path to dataset file")
    parser.add_argument('--base_model', type=str, default="NeuML/pubmedbert-base-embeddings", help="Base embedding model name")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save model and logs")
    parser.add_argument('--embedding_dim', type=int, default=128, help="Embedding dimension for the contrastive model (default: 128)")
    parser.add_argument('--input_dim', type=int, default=768, help="Input dimension for embeddings - matches dimension of base model")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--faiss_batch_size', type=int, default=256, help="Batch size for FAISS search to avoid memory overload")
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize logger
    logger = TrainingLogger(os.path.join(args.save_dir, 'logs'))
    
    # Log configuration
    logger.log_config(vars(args))
    
    # Load dataset
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
    
    # Split dataset into training and validation sets (80% train, 20% validation)
    train_data, val_data = train_test_split(list(dataset.items()), test_size=0.2, random_state=42)

    # Convert train_data and val_data back to dictionary format
    train_dataset_dict = dict(train_data)
    val_dataset_dict = dict(val_data)
    
    # Create datasets and dataloaders
    train_dataset = ContrastiveQADataset(train_dataset_dict, split='train')
    val_dataset = ContrastiveQADataset(val_dataset_dict, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    

    # Load the embedding model
    embedding_model = SentenceTransformer(args.base_model)

    # Initialize the contrastive autoencoder configuration
    config = ContrastiveAutoencoderConfig(input_dim=args.input_dim, embedding_dim=args.embedding_dim)

    # Initialize the contrastive autoencoder model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveAutoencoder(config=config, embedding_model=embedding_model)
    model.to(device)
    
    # Initialize training components
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    criterion = TripletLoss(margin=0.2)
    
    # Train model
    trained_model = train_contrastive_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=model.device,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        logger=logger,
        save_dir=args.save_dir,
        faiss_batch_size=args.faiss_batch_size
    )

if __name__ == "__main__":
    main()