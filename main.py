import os
import sys
import logging
import pickle
from backend.ncf_model import HybridMovieRecommender
from backend.questionnaire import get_user_preferences

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Initializing HybridMovieRecommender...")
        recommender = HybridMovieRecommender()

        model_filepath = 'my_recommender_model.h5'
        user_encoder_filepath = 'user_encoder.pkl'
        movie_encoder_filepath = 'movie_encoder.pkl'
        history_filepath = os.path.join('FilmGenie', 'backend', 'data', 'training_history.pkl')

        # Try to load an existing model and encoders
        if os.path.exists(model_filepath):
            logger.info("Loading existing model...")
            recommender.load_model(model_filepath)
            if os.path.exists(user_encoder_filepath):
                with open(user_encoder_filepath, 'rb') as f:
                    recommender.user_encoder = pickle.load(f)
            if os.path.exists(movie_encoder_filepath):
                with open(movie_encoder_filepath, 'rb') as f:
                    recommender.movie_encoder = pickle.load(f)
            if os.path.exists(history_filepath):
                with open(history_filepath, 'rb') as f:
                    history = pickle.load(f)
                    logger.info("Model Performance Metrics:")
                    logger.info(f"Training Loss: {history['loss'][-1]:.4f}")
                    logger.info(f"Validation Loss: {history['val_loss'][-1]:.4f}")
                    logger.info(f"Training Accuracy: {history['accuracy'][-1]:.4f}")
                    logger.info(f"Validation Accuracy: {history['val_accuracy'][-1]:.4f}")
                    logger.info(f"Training Precision: {history['precision'][-1]:.4f}")
                    logger.info(f"Validation Precision: {history['val_precision'][-1]:.4f}")
                    logger.info(f"Training Recall: {history['recall'][-1]:.4f}")
                    logger.info(f"Validation Recall: {history['val_recall'][-1]:.4f}")
        else:
            logger.info("No existing model found. Training a new model...")
            recommender.train_model(epochs=5)
            recommender.save_model(model_filepath)
            with open(user_encoder_filepath, 'wb') as f:
                pickle.dump(recommender.user_encoder, f)
            with open(movie_encoder_filepath, 'wb') as f:
                pickle.dump(recommender.movie_encoder, f)


        recommender.schedule_retraining(frequency_days=7)  # Retrain weekly

        logger.info("Collecting user preferences via questionnaire...")
        user_preferences = get_user_preferences()

        # Create a temporary user profile
        user_profile = {
            'user_id': -1,
            'preferences': user_preferences,
            'watched_movies': [],
            'ratings': []
        }
        recommender.users.append(user_profile)

        logger.info("Generating recommendations based on your preferences...")
        recommendations = recommender.recommend(user_id=-1, top_n=10)

        logger.info("\nTop 10 Movie Recommendations:")
        for i, row in recommendations.iterrows():
            logger.info(f"{i+1}. {row['Title']} (Movie ID: {row['MovieID']})")

        logger.info("Evaluating model...")
        recommender.evaluate_model()

        logger.info("Done.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    # Add the FilmGenie directory to the Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main() 