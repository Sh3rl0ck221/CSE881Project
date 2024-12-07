import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import csv

class PlaylistDataset(Dataset):
    def __init__(self, playlists, song_embeddings, split_ratio=0.6, is_test=False):
        self.playlists = playlists
        self.song_embeddings = song_embeddings
        self.split_ratio = split_ratio
        self.is_test = is_test
        
    def __len__(self):
        return len(self.playlists)
    
    def __getitem__(self, idx):
        playlist = [int(x) for x in self.playlists[idx]]
        
        if len(playlist) < 3:  # Skip too short playlists
            return None
        
        # Calculate split point
        split_point = max(1, int(len(playlist) * self.split_ratio))
        
        if self.is_test:
            # For test set, use first split_ratio as input, rest as target
            input_seq = playlist[:split_point]
            target_seq = playlist[split_point:]
            
            if not target_seq:  # Skip if no target sequence
                return None
            
            return {
                'input_ids': torch.LongTensor(input_seq),
                'target': torch.LongTensor(target_seq),
                'playlist_len': len(playlist)
            }
        else:
            # For training, use sliding window approach on first split_ratio portion
            train_seq = playlist[:split_point]
            if len(train_seq) < 3:  # Skip if sequence too short
                return None
            
            # Create input and target sequences
            input_seq = train_seq[:-1]
            target = train_seq[-1]
            
            return {
                'input_ids': torch.LongTensor(input_seq),
                'target': torch.LongTensor([target]),
                'playlist_len': len(train_seq)
            }
    
    def collate_fn(self, batch):
        # Remove None values
        batch = [x for x in batch if x is not None]
        if not batch:
            return None
        
        # Find max length in this batch
        max_input_len = max(len(x['input_ids']) for x in batch)
        
        # Prepare attention masks and pad sequences
        input_ids = []
        attention_masks = []
        targets = []
        playlist_lens = []
        
        padding_value = num_songs  # Use num_songs as padding value (since valid IDs are 0 to num_songs - 1)
        
        if self.is_test:
            max_target_len = max(len(x['target']) for x in batch)
            target_padding_value = -1  # Use -1 as padding value for targets
        else:
            target_padding_value = None  # Not needed for training
        
        for item in batch:
            input_seq = item['input_ids']
            pad_len = max_input_len - len(input_seq)
            
            # Pad input sequence
            input_ids.append(torch.cat([input_seq, torch.full((pad_len,), padding_value, dtype=torch.long)]))
            
            # Create attention mask
            attention_masks.append(torch.cat([torch.ones(len(input_seq)), torch.zeros(pad_len)]))
            
            if self.is_test:
                target_seq = item['target']
                target_pad_len = max_target_len - len(target_seq)
                # Pad target sequence with -1
                targets.append(torch.cat([target_seq, torch.full((target_pad_len,), -1, dtype=torch.long)]))
            else:
                # For training, targets are single values
                targets.append(item['target'])
            
            playlist_lens.append(item['playlist_len'])
        
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'target': torch.stack(targets) if self.is_test else torch.cat(targets),
            'playlist_len': torch.tensor(playlist_lens)
        }

class TransformerPlaylistModel(nn.Module):
    def __init__(self, num_songs, embedding_dim, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_songs = num_songs
        # Adjust num_embeddings and set padding_idx
        self.song_embeddings = nn.Embedding(num_songs + 1, embedding_dim, padding_idx=num_songs)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(embedding_dim, num_songs)  # Output size is num_songs
        
    def forward(self, input_ids, attention_mask=None):
        # Get embeddings
        x = self.song_embeddings(input_ids)
        x = self.pos_encoder(x)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            attention_mask = attention_mask == 0  # Mask positions where attention_mask == 0
        
        # Pass through transformer
        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        
        # Get predictions for last position
        x = x[:, -1, :]  # Take last sequence element
        x = self.output_layer(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, embedding_dim]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def train_model(model, train_loader, num_epochs=10, device='cuda'):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in train_bar:
            if batch is None:
                continue
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

def evaluate_model(model, test_loader, k_values=[50, 100, 200], device='cuda'):
    model.eval()
    metrics = {f'precision@{k}': [] for k in k_values}
    metrics.update({f'recall@{k}': [] for k in k_values})
    metrics.update({f'ndcg@{k}': [] for k in k_values})
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if batch is None:
                continue
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target']  # Keep targets on CPU for easier handling of variable lengths
            
            outputs = model(input_ids, attention_mask)
            
            # Get top-k predictions
            _, top_items = torch.topk(outputs, max(k_values))
            top_items = top_items.cpu()
            
            # Calculate metrics for each sequence
            for i in range(len(targets)):
                # Get non-padded targets for this sequence
                target_seq = targets[i]
                true_items = set(target_seq[target_seq >= 0].numpy())
                if not true_items:  # Skip if no targets
                    continue
                    
                pred_items = top_items[i].numpy()
                
                for k in k_values:
                    pred_k = set(pred_items[:k])
                    
                    # Precision@k
                    precision = len(pred_k & true_items) / k
                    metrics[f'precision@{k}'].append(precision)
                    
                    # Recall@k
                    recall = len(pred_k & true_items) / len(true_items)
                    metrics[f'recall@{k}'].append(recall)
                    
                    # NDCG@k
                    dcg = sum([1 / np.log2(j + 2) for j, item in enumerate(pred_items[:k]) 
                             if item in true_items])
                    idcg = sum([1 / np.log2(j + 2) for j in range(min(k, len(true_items)))])
                    ndcg = dcg / idcg if idcg > 0 else 0
                    metrics[f'ndcg@{k}'].append(ndcg)
    
    # Average metrics
    return {k: np.mean(v) for k, v in metrics.items()}

def load_playlists(file_path, num_playlists=1000):
    playlists = []
    with open(file_path, mode="r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            playlists.append(row)
    return playlists[:num_playlists]

def main():
    global num_songs  # Make num_songs global so it can be accessed in Dataset
    # Load data
    lines = load_playlists('../processed_data/playlists_seq.csv', num_playlists=1000)

    song_embeddings = np.load('song_emb.npy')
    num_songs = song_embeddings.shape[0]
    embedding_dim = song_embeddings.shape[1]
    
    # Create datasets
    train_dataset = PlaylistDataset(lines, song_embeddings, split_ratio=0.6, is_test=False)
    test_dataset = PlaylistDataset(lines, song_embeddings, split_ratio=0.6, is_test=True)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=128, 
        shuffle=True, 
        collate_fn=train_dataset.collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=128, 
        collate_fn=test_dataset.collate_fn
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerPlaylistModel(num_songs, embedding_dim).to(device)
    
    # Initialize embedding layer with pretrained embeddings
    with torch.no_grad():
        # Create a new embedding matrix with an extra row for padding
        new_song_embeddings = np.vstack([song_embeddings, np.zeros((1, embedding_dim))])
        model.song_embeddings.weight.copy_(torch.from_numpy(new_song_embeddings))
    
    # Train model
    train_model(model, train_loader, num_epochs=50, device=device)
    
    # Save final model
    torch.save(model.state_dict(), '../models/final_model_seq.pt')
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device=device)
    
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
