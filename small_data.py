import json
import pandas as pd
from pathlib import Path

def extract_song_data(json_file):
    # Read the JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Lists to store the extracted data
    songs = []
    song_id = 0  # Counter for unique song IDs
    seen_songs = {}  # Dictionary to track unique songs (by track_uri)
    popularity_count = {}  # Dictionary to track number of playlist appearances
    
    # Process each playlist
    for playlist in data['playlists']:
        for track in playlist['tracks']:
            # Use track_uri as the unique identifier
            song_key = track['track_uri']
            
            # Update popularity count
            if song_key not in popularity_count:
                popularity_count[song_key] = 1
            else:
                popularity_count[song_key] += 1
            
            # Only add the song if we haven't seen it before
            if song_key not in seen_songs:
                songs.append({
                    'song_id': song_id,
                    'artist_name': track['artist_name'],
                    'album_name': track['album_name'],
                    'track_name': track['track_name'],
                    'track_uri': track['track_uri'],
                    'popularity_count': popularity_count[song_key]
                })
                seen_songs[song_key] = song_id
                song_id += 1
            else:
                # Update popularity count for existing songs
                song_index = next(i for i, song in enumerate(songs) if song['song_id'] == seen_songs[song_key])
                songs[song_index]['popularity_count'] = popularity_count[song_key]
    
    # Create DataFrame
    df = pd.DataFrame(songs)
    
    # Save to CSV
    output_dir = "CSE_881/CSE881Project/processed_data/small_data"
    output_file = Path(output_dir).with_suffix('.csv')
    df.to_csv(output_file, index=False)
    print(f"Saved {len(songs)} unique songs to {output_file}")
    
    # Print some statistics
    print(f"\nTotal unique songs: {len(songs)}")
    print("\nMost popular songs:")
    print(df.nlargest(5, 'popularity_count')[['track_name', 'artist_name', 'popularity_count']])
    print("\nFirst few entries:")
    print(df.head())

def main():
    json_file = "CSE_881/CSE881Project/data/data/mpd.slice.0-999.json"  # Update this path
    extract_song_data(json_file)

if __name__ == "__main__":
    main()