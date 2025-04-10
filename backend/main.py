import json
import os
import logging
import numpy as np
import pandas as pd
from questionnaire import MovieQuestionnaire
from ncf_model import MovieRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_fake_users():
    """Load fake user data for training"""
    try:
        fake_users_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_users.json')
        with open(fake_users_path, 'r') as f:
            data = json.load(f)
            return data.get('users', [])  # Get the users array from the JSON
    except Exception as e:
        logger.error(f"Error loading fake users: {str(e)}")
        return []

def main():
    try:
        # Initialize components
        questionnaire = MovieQuestionnaire()
        recommender = MovieRecommender(load_ratings_data=False)  # Don't load ratings data
        
        # Step 1: Run questionnaire
        print("\n=== Welcome to FilmGenie Movie Recommendation System ===\n")
        print("Let's find the perfect movie for you! Please answer a few questions about your preferences.\n")
        
        # Get user preferences
        user_preferences = questionnaire.get_user_preferences()
        
        # Display collected preferences
        print("\n=== Your Preferences ===")
        for key, value in user_preferences.items():
            if isinstance(value, list):
                print(f"{key}: {', '.join(value)}")
            else:
                print(f"{key}: {value}")
        
        # Step 2: Load training data
        print("\nLoading training data...")
        fake_users = load_fake_users()
        if not fake_users:
            logger.error("No training data available")
            return
        
        # Step 3: Train the model
        print("\nTraining the recommendation model...")
        try:
            # Process user preferences into feature vectors
            user_vectors = []
            for user in fake_users:
                # Create user data in the expected format
                user_data = {
                    'preferences': user['preferences'],
                    'watched_movies': user['watched_movies'],
                    'ratings': user['ratings']
                }
                user_vector = recommender.create_user_profile(user_data)
                if user_vector is not None:
                    user_vectors.append(user_vector)
            
            # Convert to numpy array
            training_data = np.array(user_vectors)
            
            # Train the model
            recommender.train(training_data, epochs=5)
            print("Model training completed!")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return
        
        # Step 4: Generate recommendations
        print("\nGenerating personalized recommendations...")
        try:
            # Create user profile from questionnaire responses
            user_data = {
                'preferences': user_preferences,
                'watched_movies': [],  # Start with empty watched movies
                'ratings': []          # Start with empty ratings
            }
            
            # Get user profile vector
            user_profile = recommender.create_user_profile(user_data)
            
            # Get recommendations
            recommendations = recommender.get_recommendations(user_profile, top_n=10)
            
            # Display recommendations
            print("\n=== Your Personalized Movie Recommendations ===")
            for i, (movie_id, score) in enumerate(recommendations, 1):
                movie_info = recommender.movies[recommender.movies['movieId'] == movie_id].iloc[0]
                print(f"\n{i}. {movie_info['title']}")
                print(f"   Genres: {', '.join(movie_info['genres'])}")
                print(f"   Match Score: {score:.2f}")
                
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 