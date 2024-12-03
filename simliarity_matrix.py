import pandas as pd
from collections import defaultdict
from Levenshtein import ratio
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity_matrix(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Print matrix dimensions right at the start
    n_songs = len(df)
    print(f"\nMatrix dimensions: {n_songs} x {n_songs}")
    print(f"Maximum possible entries (excluding diagonal): {(n_songs * (n_songs - 1)) // 2:,}")
    
    # Initialize similarity matrix using defaultdict to handle missing keys
    S = defaultdict(int)
    
    # Calculate similarity based on artist_name
    artist_dict = defaultdict(list)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        artist_dict[row['artist_name']].append(row['song_id'])
    
    # Calculate similarity scores for artists
    for artist_name, song_ids in tqdm(artist_dict.items(), total=len(artist_dict)):
        for i, id1 in enumerate(song_ids):
            for id2 in song_ids[i+1:]:  # Only compare with songs we haven't seen yet
                key = f"{min(id1, id2)} {max(id1, id2)}"
                S[key] += 1

    # Calculate similarity based on album_name
    album_dict = defaultdict(list)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        album_dict[row['album_name']].append(row['song_id'])
    
    # Calculate similarity scores for albums
    for album_name, song_ids in tqdm(album_dict.items(), total=len(album_dict)):
        for i, id1 in enumerate(song_ids):
            for id2 in song_ids[i+1:]:  # Only compare with songs we haven't seen yet
                key = f"{min(id1, id2)} {max(id1, id2)}"
                S[key] += 1

    # Calculate density before track name comparison
    n_songs = len(df)
    max_possible_pairs = (n_songs * (n_songs - 1)) // 2  # Total possible pairs in upper triangle
    current_pairs = len(S)
    density_before = current_pairs / max_possible_pairs
    print(f"\nBefore track name comparison:")
    print(f"Number of songs: {n_songs}")
    print(f"Maximum possible pairs: {max_possible_pairs}")
    print(f"Current number of pairs: {current_pairs}")
    print(f"Current density: {density_before:.4%}")

    # Replace the track name comparison section with:
    print("\nCalculating TF-IDF similarities...")
    
    # Prepare track names list and id mapping
    track_names_list = df['track_name'].tolist()
    track_ids = df['song_id'].tolist()
    
    # Calculate TF-IDF similarities
    similarities = tfidf_similarity(track_names_list)
    print(similarities)
    
    # Convert similarities to our dictionary format using vectorized operations
    print("\nProcessing similarity pairs...")
    
    # Initialize new_pairs and count_2 counters
    new_pairs = 0
    count_2 = 0
    
    # Get indices where similarity >= 0.5 using numpy's where
    print("\nFinding high similarity pairs...")
    # Create a mask with progress bar
    mask = np.zeros_like(similarities, dtype=bool)
    for i in tqdm(range(similarities.shape[0]), desc="Creating similarity mask"):
        for j in range(i + 1, similarities.shape[1]):  # Only upper triangle
            if similarities[i, j] >= 0.5:
                mask[i, j] = True
    
    high_sim_indices = np.where(mask)
    
    count_1 = (len(track_names_list) * (len(track_names_list) - 1)) // 2  # Total possible pairs
    count_2 = 0
    
    print("\nAdding high-similarity pairs...")
    for i, j in tqdm(zip(*high_sim_indices), total=len(high_sim_indices[0])):
        key = f"{min(track_ids[i], track_ids[j])} {max(track_ids[i], track_ids[j])}"
        if key not in S:
            new_pairs += 1
        S[key] += similarities[i, j]
        # Increment count_2 only if the similarity score is greater than 1
        if S[key] > 1:
            count_2 += 1

    # Calculate final density
    final_pairs = len(S)
    final_density = final_pairs / max_possible_pairs
    
    print(f"\nAfter track name comparison:")
    print(f"New pairs added: {new_pairs}")
    print(f"Final number of pairs: {final_pairs}")
    print(f"Final density: {final_density:.4%}")
    print(f"Change in density: {(final_density - density_before):.4%}")
    
    # Update count_1 to be the actual number of non-zero similarities
    count_1 = len(S)  # This is the same as final_pairs
    
    return S, n_songs, count_1, count_2

def save_similarity_matrix(S, output_path, format='pkl'):
    """Save the similarity matrix to a file."""
    if format == 'pkl':
        import pickle
        with open(f"{output_path}.pkl", 'wb') as f:
            pickle.dump(S, f)
    elif format == 'npy':
        import numpy as np
        # Convert defaultdict to regular dictionary
        S_dict = dict(S)
        np.save(f"{output_path}.npy", S_dict)

def dict_to_sparse_matrix(S, n_songs):
    """
    Convert similarity dictionary to sparse matrix format.
    
    Args:
        S: Dictionary with keys as 'id1 id2' and values as similarity scores
        n_songs: Total number of songs (matrix dimension)
    
    Returns:
        scipy.sparse.csr_matrix: Sparse similarity matrix
    """
    # Create lists for row indices, column indices, and values
    rows = []
    cols = []
    values = []
    
    # Parse dictionary entries
    for key, value in S.items():
        id1, id2 = map(int, key.split())
        # Add both (i,j) and (j,i) since it's symmetric
        rows.extend([id1, id2])
        cols.extend([id2, id1])
        values.extend([value, value])
    
    # Create sparse matrix
    sparse_matrix = csr_matrix((values, (rows, cols)), 
                             shape=(n_songs, n_songs))
    
    return sparse_matrix

def tfidf_similarity(tracks):
    """
    Calculate similarity using TF-IDF and cosine similarity
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tracks)
    return cosine_similarity(tfidf_matrix)

def main():
    # Example usage
    csv_file = "CSE_881/CSE881Project/processed_data/small_data.csv"
    similarity_dict, n_songs, count_1, count_2 = calculate_similarity_matrix(csv_file)
    
    # Convert to sparse matrix
    sparse_sim_matrix = dict_to_sparse_matrix(similarity_dict, n_songs)
    
    # Print matrix information
    print("\nSparse Matrix Information:")
    print(f"Shape: {sparse_sim_matrix.shape}")
    print(f"Number of non-zero elements: {sparse_sim_matrix.nnz:,}")
    print(f"Memory usage: {sparse_sim_matrix.data.nbytes / 1024 / 1024:.2f} MB")
    
    # Optional: Save sparse matrix
    from scipy.sparse import save_npz
    save_npz("CSE_881/CSE881Project/processed_data/similarity_matrix_sparse.npz", 
             sparse_sim_matrix)
    
    # Save the similarity matrix
    output_path = "CSE_881/CSE881Project/processed_data/similarity_matrix"
    save_similarity_matrix(similarity_dict, output_path, format='pkl')  # or format='npy'
    
    # Print some example similarities
    print(f"Total number of similarities: {count_1}")
    print(f"Total number of similarities greater than 1: {count_2}")

if __name__ == "__main__":
    main()
