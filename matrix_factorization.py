import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, lil_matrix
import csv
from tqdm import tqdm

class MatrixFactorization(nn.Module):
    def __init__(self, num_playlists, num_songs, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.playlist_embeddings = nn.Embedding(num_playlists, embedding_dim)
        self.song_embeddings = nn.Embedding(num_songs, embedding_dim)
        
    def forward(self, playlist_ids, song_ids):
        playlist_embeds = self.playlist_embeddings(playlist_ids)
        song_embeds = self.song_embeddings(song_ids)
        return (playlist_embeds * song_embeds).sum(dim=1)

def prepare_data_matrix(lines):
    # Get dimensions
    num_playlists = len(lines)
    num_songs = max(max(int(song_id) for song_id in playlist) for playlist in lines) + 1
    
    # Create LIL matrix for efficient construction
    print("Creating interaction matrix...")
    interaction_matrix = lil_matrix((num_playlists, num_songs), dtype=np.float32)
    
    # Fill the matrix with progress bar
    for playlist_idx in tqdm(range(len(lines)), desc="Building interaction matrix"):
        song_indices = [int(song_id) for song_id in lines[playlist_idx]]
        interaction_matrix[playlist_idx, song_indices] = 1
    
    print("Converting to CSR format...")
    interaction_matrix = interaction_matrix.tocsr()
    
    return interaction_matrix, num_playlists, num_songs

def train_test_split_matrix(interaction_matrix, test_size=0.4):
    """Split interaction matrix into train and test"""
    print("Splitting into train and test sets...")
    
    # Convert to LIL format for efficient row modifications
    train_matrix = interaction_matrix.tolil()
    test_matrix = lil_matrix(interaction_matrix.shape)
    
    # For each playlist, move some interactions to test set
    for playlist_idx in tqdm(range(interaction_matrix.shape[0]), desc="Splitting data"):
        playlist_songs = interaction_matrix[playlist_idx].nonzero()[1]
        if len(playlist_songs) > 0:
            n_test = max(1, int(len(playlist_songs) * test_size))
            test_songs = np.random.choice(playlist_songs, n_test, replace=False)
            
            # Remove test songs from train matrix
            train_matrix[playlist_idx, test_songs] = 0
            
            # Add test songs to test matrix
            test_matrix[playlist_idx, test_songs] = 1
    
    print("Converting matrices to CSR format...")
    return train_matrix.tocsr(), test_matrix.tocsr()

def train_matrix_factorization(model, train_matrix, num_epochs=5, batch_size=1024):
    # hyperparameter tuning
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    # Convert sparse matrix to indices of non-zero elements
    playlist_indices, song_indices = train_matrix.nonzero()
    
    # Create epoch progress bar
    epoch_bar = tqdm(range(num_epochs), desc="Training Epochs")
    
    for epoch in epoch_bar:
        model.train()
        total_loss = 0
        
        # Shuffle indices
        indices = np.arange(len(playlist_indices))
        np.random.shuffle(indices)
        
        # Create batch progress bar
        batch_bar = tqdm(range(0, len(indices), batch_size), 
                        desc=f"Epoch {epoch + 1}", 
                        leave=False)
        
        for start_idx in batch_bar:
            end_idx = min(start_idx + batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            # Positive samples
            playlist_ids = torch.LongTensor(playlist_indices[batch_indices])
            song_ids = torch.LongTensor(song_indices[batch_indices])
            
            # Generate negative samples
            neg_song_ids = torch.LongTensor(
                np.random.randint(0, train_matrix.shape[1], size=len(batch_indices)))

            # Forward pass
            pos_pred = model(playlist_ids, song_ids)
            neg_pred = model(playlist_ids, neg_song_ids)
            
            # Loss computation
            loss = criterion(pos_pred, torch.ones_like(pos_pred)) + \
                   criterion(neg_pred, torch.zeros_like(neg_pred))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update batch progress bar
            batch_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / (len(indices) // batch_size)
        epoch_bar.set_postfix({'avg_loss': f"{avg_loss:.4f}"})

def evaluate_recommendations(model, test_matrix, k_values=[100, 200, 500]):
    model.eval()
    metrics = {f'precision@{k}': [] for k in k_values}
    metrics.update({f'recall@{k}': [] for k in k_values})
    metrics.update({f'ndcg@{k}': [] for k in k_values})
    
    # Convert test matrix to LIL format for efficient row access
    test_matrix = test_matrix.tolil()
    
    # Create progress bar for evaluation
    eval_bar = tqdm(range(test_matrix.shape[0]), desc="Evaluating")
    
    with torch.no_grad():
        for playlist_idx in eval_bar:
            true_songs = set(test_matrix.rows[playlist_idx])
            if not true_songs:
                continue
            
            # Get recommendations
            playlist_id = torch.LongTensor([playlist_idx])
            all_song_ids = torch.arange(test_matrix.shape[1])
            scores = model(playlist_id.repeat(len(all_song_ids)), all_song_ids)
            
            # Get top K recommendations
            _, top_items = torch.topk(scores, max(k_values))
            recommended = top_items.cpu().numpy()
            
            # Calculate metrics for different k values
            current_metrics = {}
            for k in k_values:
                rec_k = set(recommended[:k])
                
                # Precision@k
                precision = len(rec_k & true_songs) / k
                metrics[f'precision@{k}'].append(precision)
                current_metrics[f'P@{k}'] = f"{precision:.3f}"
                
                # Recall@k
                recall = len(rec_k & true_songs) / len(true_songs)
                metrics[f'recall@{k}'].append(recall)
                current_metrics[f'R@{k}'] = f"{recall:.3f}"
                
                # NDCG@k
                dcg = sum([1 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) 
                          if item in true_songs])
                idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(true_songs)))])
                ndcg = dcg / idcg if idcg > 0 else 0
                metrics[f'ndcg@{k}'].append(ndcg)
                current_metrics[f'N@{k}'] = f"{ndcg:.3f}"
            
            # Update progress bar with current metrics
            eval_bar.set_postfix(current_metrics)
    
    # Average metrics
    return {k: np.mean(v) for k, v in metrics.items()}

def save_embeddings(song_embeddings, file_path="embeddings"):
    """
    Save embeddings in multiple formats
    """
    np.save(f"{file_path}.npy", song_embeddings)
    print(f"Embeddings saved to {file_path}.npy")

def load_embeddings(file_path="embeddings", format='npy'):
    """
    Load embeddings from disk
    """
    if format == 'npy':
        return np.load(f"{file_path}.npy")
    elif format == 'pt':
        return torch.load(f"{file_path}.pt").numpy()
    else:
        raise ValueError(f"Unknown format: {format}")


def main():
    # Path to the CSV file
    file_path = "processed_data/playlists_seq.csv"

    # Open the file and read line by line
    lines = []
    with open(file_path, mode="r") as file:
        csv_reader = csv.reader(file)
        
        # Iterate over each line
        for row in csv_reader:
            lines.append(row)  # Each row is a list of values
    num_playlists = 1000
    lines = lines[:num_playlists]

    # Prepare data
    interaction_matrix, num_playlists, num_songs = prepare_data_matrix(lines)
    train_matrix, test_matrix = train_test_split_matrix(interaction_matrix)
    
    print(f"\nDataset statistics:")
    print(f"Number of playlists: {num_playlists}")
    print(f"Number of songs: {num_songs}")
    print(f"Train matrix density: {train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]):.6f}")
    print(f"Test matrix density: {test_matrix.nnz / (test_matrix.shape[0] * test_matrix.shape[1]):.6f}\n")
    
    # Initialize model
    embedding_dim = 64
    model = MatrixFactorization(num_playlists, num_songs, embedding_dim)
    
    # Train model
    train_matrix_factorization(model, train_matrix, num_epochs=10)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_recommendations(model, test_matrix)
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Get final embeddings
    with torch.no_grad():
        song_embeddings = model.song_embeddings.weight.cpu().detach().numpy()
        playlist_embeddings = model.playlist_embeddings.weight.cpu().detach().numpy()
    
    return model, song_embeddings, playlist_embeddings, metrics

# Run the main function
model, song_embeddings, playlist_embeddings, metrics = main()
save_embeddings(song_embeddings, 'song_emb')
save_embeddings(playlist_embeddings, 'playlist_emb')
