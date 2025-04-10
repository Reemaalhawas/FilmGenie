import os
import sys
import pandas as pd
import logging
from backend.generate_sample_data import generate_sample_data

# Set project root (this file is in FilmGenie folder)
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
    """Test the recommendation model with sample data"""
    try:
        logger.info("Starting test_model...")
        logger.info(f"Project root: {PROJECT_ROOT}")
        
        # Generate sample user data
        logger.info("Generating sample user data...")
        generate_sample_data()
        
        # Initialize recommender
        recommender = MovieRecommender()
        
        # Load data
        logger.info("Loading ratings data...")
        ratings_path = os.path.join(PROJECT_ROOT, 'data', 'ratings_updated.csv')
        ratings_df = pd.read_csv(ratings_path)
        
        # Rename columns if needed
        if 'UserID' in ratings_df.columns and 'MovieID' in ratings_df.columns and 'Rating' in ratings_df.columns:
            ratings_df = ratings_df.rename(columns={
                'UserID': 'userId',
                'MovieID': 'movieId',
                'Rating': 'rating'
            })
        
        logger.info("Loading movie metadata...")
        movies_path = os.path.join(PROJECT_ROOT, 'data', 'movies (1).dat')
        movies_df = pd.DataFrame()
        with open(movies_path, 'r', encoding='latin-1') as f:
            movies_data = []
            for line in f:
                movie_id, title, genres = line.strip().split('::')
                movies_data.append({
                    'movieId': int(movie_id),
                    'title': title,
                    'genres': genres.split('|')
                })
            movies_df = pd.DataFrame(movies_data)
        
        # Load user ratings into the recommender
        logger.info("Loading user ratings into recommender...")
        recommender.load_ratings(ratings_path)
        recommender.load_movies(movies_path)
        
        # Collect user preferences
        logger.info("\nCollecting user preferences...")
        questionnaire = MovieQuestionnaire()
        user_preferences = questionnaire.get_user_preferences()
        
        # Create user profile
        logger.info("Creating user profile...")
        user_profile = {
            'preferences': user_preferences,
            'watched_movies': [],  # New user, no watched movies yet
            'ratings': []          # New user, no ratings yet
        }
        
        # Preprocess data and build model
        logger.info("Preprocessing data and building model...")
        X_user, X_movie, X_features, y = recommender.preprocess_data(ratings_df, movies_df, user_preferences)
        recommender.train([X_user, X_movie, X_features], y, epochs=5)
        
        # Get recommendations using collaborative filtering
        logger.info("\nGenerating recommendations...")
        recommendations = recommender.get_recommendations(user_profile, top_n=10)
        
        # Display recommendations
        logger.info("\nTop 10 Movie Recommendations:")
        for i, (movie_id, score) in enumerate(recommendations, 1):
            movie = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            logger.info(f"\n{i}. {movie['title']}")
            logger.info(f"   Genres: {', '.join(movie['genres'])}")
            logger.info(f"   Recommendation Score: {score:.4f}")
            logger.info(f"   Movie ID: {movie_id}")
        
        logger.info("\nTest completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in test_model: {str(e)}")
        raise

if __name__ == "__main__":
    test_model()
