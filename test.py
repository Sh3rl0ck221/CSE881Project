import pandas as pd

def compare_csv_files(old_file_path, new_file_path):
    # Read both CSV files
    old_df = pd.read_csv(old_file_path)
    new_df = pd.read_csv(new_file_path)
    
    # Drop track_url column from new_df if it exists
    if 'track_url' in new_df.columns:
        new_df = new_df.drop(columns=['track_url'])
    
    # Find records in new_df that are not in old_df
    # Using merge with specific columns to identify unique songs
    comparison = new_df.merge(
        old_df,
        on=['track_name', 'artist_name'],  # Add any other columns that define a unique song
        how='left',
        indicator=True
    ).loc[lambda x: x['_merge'] == 'left_only']
    
    # Drop the merge indicator column
    new_records = comparison.drop(columns=['_merge'])
    
    return new_records

# Example usage
old_file_path = 'CSE_881/CSE881Project/processed_data/old_small_data.csv'
new_file_path = 'CSE_881/CSE881Project/processed_data/small_data.csv'
output_file_path = 'CSE_881/CSE881Project/processed_data/new_records.csv'  # New output file path

new_records = compare_csv_files(old_file_path, new_file_path)
print(f"Number of new records: {len(new_records)}")
print("\nNew records:")
print(new_records)

# Save the new records to a CSV file
new_records.to_csv(output_file_path, index=False)
print(f"\nNew records have been saved to: {output_file_path}")