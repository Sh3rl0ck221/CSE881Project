import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import lil_matrix, save_npz
import multiprocessing as mp
from tqdm import tqdm
import math
import re
import pandas as pd

def preprocess_track_name(track_name):
    # Handle None or empty strings
    if not track_name or pd.isna(track_name):
        return "unknown"
    
    # Convert to string if not already
    track_name = str(track_name)
    
    # Remove special characters but keep letters, numbers and spaces
    track_name = re.sub(r'[^a-zA-Z0-9\s]', ' ', track_name)
    
    # Remove extra whitespace
    track_name = ' '.join(track_name.split())
    
    return track_name.lower() if track_name else "unknown"

def calculate_batch_similarities(args):
    start_idx, end_idx, tracks, batch_size = args
    local_matrix = lil_matrix((end_idx - start_idx, len(tracks)))
    
    for i in range(start_idx, end_idx):
        # Only compute upper triangle
        for j in range(i + 1, min(i + batch_size, len(tracks))):
            # Artist similarity
            artist_sim = 1 if tracks[i]["artist_name"] == tracks[j]["artist_name"] else 0
            
            # Album similarity
            album_sim = 1 if tracks[i]["album_name"] == tracks[j]["album_name"] else 0
            
            # Track name similarity
            try:
                name1 = preprocess_track_name(tracks[i]["track_name"])
                name2 = preprocess_track_name(tracks[j]["track_name"])
                
                # Skip TF-IDF if names are identical
                if name1 == name2:
                    track_name_sim = 1.0
                else:
                    vectorizer = TfidfVectorizer(
                        min_df=1,
                        stop_words=None,
                        token_pattern=r'(?u)\b\w+\b'
                    ).fit([name1, name2])
                    
                    if len(vectorizer.vocabulary_) == 0:
                        track_name_sim = 0.0  # No common terms
                    else:
                        tfidf_matrix = vectorizer.transform([name1, name2])
                        track_name_sim = cosine_similarity(tfidf_matrix)[0, 1]
            except Exception as e:
                print(f"Error processing track names: {tracks[i]['track_name']} and {tracks[j]['track_name']}")
                print(f"Error: {str(e)}")
                track_name_sim = 0.0  # Fallback similarity
            
            # Combine with weights
            sim_score = 0.5 * artist_sim + 0.3 * album_sim + 0.2 * track_name_sim
            
            # Store only if similarity is above threshold
            if sim_score > 0.1:
                local_matrix[i - start_idx, j] = sim_score
        
        # Set diagonal to 1
        if i - start_idx >= 0:
            local_matrix[i - start_idx, i] = 1
            
    return start_idx, local_matrix

if __name__ == '__main__':
    # Load JSON data
    with open("processed_data/tracks_info.json", "r") as file:
        data = json.load(file)
    
    # Extract track information
    tracks = [
        {
            "id": track["id"],
            "artist_name": track["artist_name"],
            "track_name": track["track_name"],
            "album_name": track["album_name"]
        }
        for track in data.values()
    ]

    n_tracks = len(tracks)
    batch_size = 1000
    
    # Determine number of processes and chunk size
    num_processes = mp.cpu_count() - 1  # Leave one CPU free
    chunk_size = math.ceil(n_tracks / num_processes)
    
    # Prepare arguments for each process
    process_args = [
        (i, min(i + chunk_size, n_tracks), tracks, batch_size)
        for i in range(0, n_tracks, chunk_size)
    ]
    
    # Create the final similarity matrix
    similarity_matrix = lil_matrix((n_tracks, n_tracks))
    
    print(f"Starting parallel processing with {num_processes} processes...")
    
    # Use multiprocessing to calculate similarities
    with mp.Pool(processes=num_processes) as pool:
        for start_idx, local_matrix in tqdm(
            pool.imap_unordered(calculate_batch_similarities, process_args),
            total=len(process_args)
        ):
            # Copy the local matrix to the appropriate location in the final matrix
            end_idx = min(start_idx + local_matrix.shape[0], n_tracks)
            similarity_matrix[start_idx:end_idx] = local_matrix
            
            # Mirror the upper triangle to lower triangle for symmetry
            for i in range(start_idx, end_idx):
                for j in range(i + 1, n_tracks):
                    if similarity_matrix[i, j] != 0:
                        similarity_matrix[j, i] = similarity_matrix[i, j]
    
    # Save the final matrix
    print("Saving sparse matrix...")
    save_npz('song_similarity_matrix_sparse.npz', similarity_matrix.tocsr())
    print("Sparse similarity matrix saved as 'song_similarity_matrix_sparse.npz'")
