import pandas as pd

# Define file paths
users_file = r'C:\Users\reema\FilmGenie\data\users_updated.csv'
ratings_file = r'C:\Users\reema\FilmGenie\data\ratings_updated.csv'
ratings_emotional_file = r'C:\Users\reema\FilmGenie\data\ratings_with_emotional_impact.csv'
movies_file = r'C:\Users\reema\FilmGenie\data\movies_with_actors_and_keywords.csv'

# Read CSV files
users = pd.read_csv(users_file)
ratings = pd.read_csv(ratings_file)
ratings_emotional = pd.read_csv(ratings_emotional_file)
movies = pd.read_csv(movies_file)

# Merge datasets using the correct column names
ratings_combined = pd.merge(ratings, ratings_emotional, on=['UserID', 'MovieID'], how='outer')
ratings_movies = pd.merge(ratings_combined, movies, on='MovieID', how='left')
final_merged = pd.merge(ratings_movies, users, on='UserID', how='left')

# Save the final merged dataset in chunks
output_file = r'C:\Users\reema\FilmGenie\data\final_merged_dataset.csv'
chunk_size = 10000  # Adjust the chunk size as needed

with open(output_file, 'w', encoding='utf-8', newline='') as f:
    # Write header first
    final_merged.iloc[:0].to_csv(f, index=False)
    # Write data in chunks
    for start in range(0, len(final_merged), chunk_size):
        final_merged.iloc[start:start+chunk_size].to_csv(f, index=False, header=False)

print("Merged dataset saved to:", output_file)
