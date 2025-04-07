import pandas as pd
import numpy as np
import os
import sys
import logging

# Set project root and update sys.path if necessary
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from backend.ncf_model import MovieRecommender
from backend.questionnaire import MovieQuestionnaire

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_model():
    try:
        logger.info("Starting test_model...")
        logger.info(f"Project root: {PROJECT_ROOT}")
        
        # Initialize questionnaire and recommender
        questionnaire = MovieQuestionnaire()
        recommender = MovieRecommender()
        
        # Get user preferences interactively
        logger.info("Starting questionnaire...")
        preferences = questionnaire.get_user_preferences()
        logger.info("Questionnaire completed")
        
        # Convert preferences to feature vector and DataFrame for preprocessing
        feature_vector = questionnaire.preprocess_preferences(preferences)
        user_preferences_df = pd.DataFrame([preferences])
        
        # For this demo, use empty DataFrames so that preprocess_data loads from files
        ratings_df = pd.DataFrame()
        movies_df = pd.DataFrame()
        
        logger.info("Starting data preprocessing...")
        X_user, X_movie, X_features, y = recommender.preprocess_data(ratings_df, movies_df, user_preferences_df)
        logger.info(f"Data preprocessing completed. Shapes: X_user={X_user.shape}, X_movie={X_movie.shape}, X_features={X_features.shape}, y={y.shape}")
        
        logger.info("Starting model training...")
        history = recommender.train(X_user, X_movie, X_features, y, epochs=5, batch_size=64)
        logger.info("Model training completed")
        
        logger.info("Evaluating model...")
        metrics = recommender.evaluate_model()
        logger.info("Model evaluation completed")
        
        # Test a prediction for a specific user and movie
        logger.info("Testing prediction...")
        test_user_id = 1
        test_movie_id = 1
        prediction = recommender.predict(test_user_id, test_movie_id, X_features[0])
        logger.info(f"Prediction for user {test_user_id} and movie {test_movie_id}: {prediction:.4f}")
        
        # Generate top recommendations for the test user
        logger.info("Getting recommendations...")
        recommendations = recommender.get_recommendations(test_user_id, X_features[0], top_n=5)
        logger.info("Recommendations generated successfully")
        
        # Load movies.dat to display recommendation details
        movies_data = []
        movies_file = os.path.join(PROJECT_ROOT, 'data', 'movies.dat')
        with open(movies_file, 'r', encoding='latin-1') as f:
            for line in f:
                m_id, title, genres = line.strip().split('::')
                movies_data.append({
                    'movieId': int(m_id),
                    'title': title,
                    'genres': genres
                })
        movies_df = pd.DataFrame(movies_data)
        
        print("\nHere are your personalized movie recommendations:")
        for movie_id, rating in recommendations:
            movie = movies_df[movies_df['movieId'] == movie_id]
            if not movie.empty:
                movie = movie.iloc[0]
                print(f"\nTitle: {movie['title']}")
                print(f"Predicted Rating: {rating:.2f}")
                print(f"Genres: {movie['genres']}")
                print("-" * 50)
                
    except Exception as e:
        logger.error(f"Error in test_model: {str(e)}")
        raise

if __name__ == "__main__":
    test_model()
