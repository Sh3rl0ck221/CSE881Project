from scipy.sparse import load_npz

# Load the sparse matrix
sparse_sim_matrix = load_npz("CSE_881/CSE881Project/processed_data/similarity_matrix_sparse.npz")

# Now you can use the matrix:

# Example 1: Get similarity between songs with indices 0 and 1
similarity_score = sparse_sim_matrix[0, 5]

print(similarity_score)

# Print the matrix dimensions
print(f"Matrix dimensions: {sparse_sim_matrix.shape}")

# Print number of non-zero elements
print(f"Number of stored (non-zero) elements: {sparse_sim_matrix.nnz}")

# Print the sparsity ratio (percentage of elements that are non-zero)
sparsity = sparse_sim_matrix.nnz / (sparse_sim_matrix.shape[0] * sparse_sim_matrix.shape[1])
print(f"Sparsity: {sparsity:.4%}")

# Example 2: Get all similarities for song 0
song_5_similarities = sparse_sim_matrix[5].toarray().flatten()

print(song_5_similarities)

# Example 3: If you need the full dense matrix (be careful with large matrices!)
# dense_matrix = sparse_sim_matrix.toarray()  # Only do this for small matrices!

# Example 4: Get the shape of the matrix
n_songs = sparse_sim_matrix.shape[0]
print(f"Matrix dimensions: {sparse_sim_matrix.shape}")

# Method 1: Using nonzero() to get indices of all non-zero elements
rows, cols = sparse_sim_matrix.nonzero()
first_row, first_col = rows[0], cols[0]
first_value = sparse_sim_matrix[first_row, first_col]

print(f"First non-zero element is at position ({first_row}, {first_col}) with value {first_value}")

# Method 2: If you want to find all values for a specific row
# Get all non-zero elements in row 0
row_0 = sparse_sim_matrix.getrow(0)
row_0_nonzero = row_0.nonzero()[1]  # Get indices of non-zero elements
if len(row_0_nonzero) > 0:
    first_nonzero_in_row = row_0_nonzero[0]
    value = row_0[0, first_nonzero_in_row]
    print(f"First non-zero element in row 0 is at column {first_nonzero_in_row} with value {value}")