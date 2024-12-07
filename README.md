# CSE881Project
Repo for CSE 881 Projecy

# Model Training Instructions

Follow the instructions below to set up your directory and run the code.

## Directory Structure

Before running the code, ensure your directory is organized as follows:
```
project_root/ 
├── code/ 
│ ├── matrix_factorization.py 
│ ├── seq.py 
│ ├── graph_seq.py 
│ ├── textual_graph_seq.py 
├── processed_data/ 
│ ├── playlists_seq.csv 
│ ├── similarity_matrix.pkl 
│ ├── similarity_matrix_sparse.npz
```
### Files Overview
- **code/**: Contains the scripts for different models:
  - `matrix_factorization.py`: Code for matrix factorization.
  - `seq.py`: Sequential modeling with transformer-based architecture.
  - `graph_seq.py`: Graph-based sequential model.
  - `textual_graph_seq.py`: Graph-based sequential model with textual augmentation.
- **processed_data/**: Contains the preprocessed input data:
  - `playlists_seq.csv`: Sequential playlist data.
  - `similarity_matrix.pkl`: Precomputed song similarity matrix.
  - `similarity_matrix_sparse.npz`: Sparse representation of the similarity matrix.

## Running the Code

### Step 1: Run Matrix Factorization
Start by running the matrix factorization code to generate initial song and playlist embeddings.

```bash
python code/matrix_factorization.py
```
### Step 2: Run Other Models
The generated embeddings (song_emb.npy and playlist_emb.npy) are required for the subsequent models. 

You can now run any of the other scripts:

Sequential Model:
```bash
python code/seq.py
```

Graph-based Sequential Model:
```bash
python code/graph_seq.py
```

Textual Graph-based Sequential Model:
```bash
python code/textual_graph_seq.py
```

