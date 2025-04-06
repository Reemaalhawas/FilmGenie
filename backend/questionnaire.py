import pandas as pd
import numpy as np
from ncf_model import MovieRecommender

class MovieQuestionnaire:
    def __init__(self):
        self.questions = {
            'mood': {
                'question': 'What mood are you in right now?',
                'options': ['Happy', 'Sad', 'Energetic', 'Thoughtful', 'Excited', 'Calm', 'Stressed']
            },
            'desired_mood': {
                'question': 'How do you want the movie to make you feel?',
                'options': ['Happy', 'Inspired', 'Excited', 'Thoughtful', 'Relaxed', 'Thrilled']
            },
            'genres_liked': {
                'question': 'Which genres do you enjoy the most? (Select multiple)',
                'options': ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Romance', 'Thriller', 'Horror', 'Documentary']
            },
            'genres_disliked': {
                'question': 'Are there any genres you dislike or want to avoid? (Select multiple)',
                'options': ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Romance', 'Thriller', 'Horror', 'Documentary']
            },
            'attention_level': {
                'question': 'How much attention do you want to give this movie?',
                'options': ['Casual (can multitask)', 'Moderate (some focus needed)', 'Full attention required']
            },
            'complexity_preference': {
                'question': 'How do you feel about mind-bending plots?',
                'options': ['Prefer simple plots', 'Like some complexity', 'Love complex mind-bending stories']
            },
            'pacing_preference': {
                'question': 'What pacing do you prefer?',
                'options': ['Slow and steady', 'Moderate pace', 'Fast-paced']
            },
            'movie_length_preference': {
                'question': 'How do you feel about long movies (2.5+ hours)?',
                'options': ['Prefer shorter movies', 'Don\'t mind longer movies', 'Love epic length movies']
            },
            'language_preference': {
                'question': 'Do you prefer movies from a specific language?',
                'options': ['English', 'Foreign language', 'No preference']
            },
            'actor_preference': {
                'question': 'Do you have a favorite actor or director whose movies you want to see?',
                'options': ['Yes, specific actor/director', 'No preference']
            },
            'director_preference': {
                'question': 'Do you have a favorite director?',
                'options': ['Yes, specific director', 'No preference']
            },
            'time_period_preference': {
                'question': 'Do you prefer movies from a specific time period?',
                'options': ['Classic (pre-1990)', 'Modern (1990-present)', 'No preference']
            },
            'rating_preference': {
                'question': 'Would you like recommendations based on highly-rated movies?',
                'options': ['Yes, highly rated only', 'No preference']
            },
            'streaming_preference': {
                'question': 'Which streaming platforms do you use?',
                'options': ['Netflix', 'Amazon Prime', 'Hulu', 'Disney+', 'Other', 'No preference']
            }
        }
        
    def get_user_preferences(self):
        preferences = {}
        print("\nWelcome to FilmGenie! Please answer the following questions to get personalized movie recommendations.\n")
        
        for key, q in self.questions.items():
            print(f"\n{q['question']}")
            for i, option in enumerate(q['options'], 1):
                print(f"{i}. {option}")
            
            if 'genres' in key:  # Handle multiple selection for genres
                while True:
                    try:
                        selections = input("\nEnter numbers separated by commas (e.g., 1,2,3): ").split(',')
                        selections = [int(s.strip()) for s in selections]
                        if all(1 <= s <= len(q['options']) for s in selections):
                            preferences[key] = [q['options'][s-1] for s in selections]
                            break
                        print("Invalid selection. Please try again.")
                    except ValueError:
                        print("Invalid input. Please enter numbers separated by commas.")
            else:
                # Handle single selection
                while True:
                    try:
                        selection = int(input("\nEnter your choice (number): "))
                        if 1 <= selection <= len(q['options']):
                            preferences[key] = q['options'][selection-1]
                            break
                        print("Invalid selection. Please try again.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
        
        return preferences
    
    def preprocess_preferences(self, preferences):
        # Convert preferences to numerical features
        feature_vector = []
        
        # Mood encoding (one-hot)
        mood_encoding = {mood: i for i, mood in enumerate(self.questions['mood']['options'])}
        feature_vector.append(mood_encoding[preferences['mood']])
        
        # Desired mood encoding (one-hot)
        desired_mood_encoding = {mood: i for i, mood in enumerate(self.questions['desired_mood']['options'])}
        feature_vector.append(desired_mood_encoding[preferences['desired_mood']])
        
        # Genres (multi-hot)
        genre_encoding = {genre: i for i, genre in enumerate(self.questions['genres_liked']['options'])}
        # Create a binary vector for liked genres
        liked_genres_vector = [0] * len(self.questions['genres_liked']['options'])
        for genre in preferences['genres_liked']:
            liked_genres_vector[genre_encoding[genre]] = 1
        feature_vector.extend(liked_genres_vector)
        
        # Create a binary vector for disliked genres
        disliked_genres_vector = [0] * len(self.questions['genres_disliked']['options'])
        for genre in preferences['genres_disliked']:
            disliked_genres_vector[genre_encoding[genre]] = 1
        feature_vector.extend(disliked_genres_vector)
        
        # Attention level
        attention_encoding = {level: i for i, level in enumerate(self.questions['attention_level']['options'])}
        feature_vector.append(attention_encoding[preferences['attention_level']])
        
        # Complexity preference
        complexity_encoding = {level: i for i, level in enumerate(self.questions['complexity_preference']['options'])}
        feature_vector.append(complexity_encoding[preferences['complexity_preference']])
        
        # Pacing preference
        pacing_encoding = {pace: i for i, pace in enumerate(self.questions['pacing_preference']['options'])}
        feature_vector.append(pacing_encoding[preferences['pacing_preference']])
        
        # Movie length preference
        length_encoding = {length: i for i, length in enumerate(self.questions['movie_length_preference']['options'])}
        feature_vector.append(length_encoding[preferences['movie_length_preference']])
        
        # Language preference
        language_encoding = {lang: i for i, lang in enumerate(self.questions['language_preference']['options'])}
        feature_vector.append(language_encoding[preferences['language_preference']])
        
        # Actor preference
        actor_encoding = {pref: i for i, pref in enumerate(self.questions['actor_preference']['options'])}
        feature_vector.append(actor_encoding[preferences['actor_preference']])
        
        # Director preference
        director_encoding = {pref: i for i, pref in enumerate(self.questions['director_preference']['options'])}
        feature_vector.append(director_encoding[preferences['director_preference']])
        
        # Time period preference
        time_encoding = {period: i for i, period in enumerate(self.questions['time_period_preference']['options'])}
        feature_vector.append(time_encoding[preferences['time_period_preference']])
        
        # Rating preference
        rating_encoding = {pref: i for i, pref in enumerate(self.questions['rating_preference']['options'])}
        feature_vector.append(rating_encoding[preferences['rating_preference']])
        
        # Streaming preference
        streaming_encoding = {platform: i for i, platform in enumerate(self.questions['streaming_preference']['options'])}
        feature_vector.append(streaming_encoding[preferences['streaming_preference']])
        
        return np.array(feature_vector, dtype=np.float32)

def main():
    try:
        # Initialize questionnaire and recommender
        questionnaire = MovieQuestionnaire()
        recommender = MovieRecommender()
        
        # Get user preferences
        print("\nWelcome to FilmGenie - Your Personal Movie Recommender!")
        preferences = questionnaire.get_user_preferences()
        
        # Preprocess preferences
        feature_vector = questionnaire.preprocess_preferences(preferences)
        
        # Create empty DataFrames for initial preprocessing
        # The actual data will be loaded in the preprocess_data method
        ratings_df = pd.DataFrame()
        movies_df = pd.DataFrame()
        
        # Train the model
        print("\nTraining the recommendation model...")
        print("Loading data files...")
        print("Checking ratings file structure...")
        ratings_df = pd.read_csv('../data/ratings_updated.csv')
        print(f"Ratings file columns: {ratings_df.columns.tolist()}")
        print(f"Number of ratings: {len(ratings_df)}")
        
        X_user, X_movie, X_features, y = recommender.preprocess_data(ratings_df, movies_df, pd.DataFrame([preferences]))
        recommender.train(X_user, X_movie, X_features, y)
        
        # Get recommendations
        print("\nGenerating personalized recommendations...")
        recommendations = recommender.get_recommendations(0, feature_vector)  # Using 0 as temporary user_id
        
        # Load movies data for display
        movies_data = []
        with open('../data/movies.dat', 'r', encoding='latin-1') as f:
            for line in f:
                movie_id, title, genres = line.strip().split('::')
                movies_data.append({
                    'movieId': int(movie_id),
                    'title': title,
                    'genres': genres
                })
        movies_df = pd.DataFrame(movies_data)
        
        # Display recommendations
        print("\nHere are your personalized movie recommendations:")
        for movie_id, rating in recommendations:
            movie = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            print(f"\nTitle: {movie['title']}")
            print(f"Predicted Rating: {rating:.2f}")
            print(f"Genres: {movie['genres']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please make sure all required data files are present in the data directory.")
        print("Required files:")
        print("- data/movies.dat")
        print("- data/ratings_updated.csv")
        print("\nDebug information:")
        try:
            ratings_df = pd.read_csv('../data/ratings_updated.csv')
            print(f"Ratings file columns: {ratings_df.columns.tolist()}")
            print(f"Number of ratings: {len(ratings_df)}")
        except Exception as e2:
            print(f"Error reading ratings file: {str(e2)}")

if __name__ == "__main__":
    main() 