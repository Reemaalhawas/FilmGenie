# api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import json
import pickle
import logging
from backend.ncf_model import HybridMovieRecommender
from backend.questionnaire import get_user_preferences

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the recommender
logger.info("Initializing HybridMovieRecommender...")
recommender = HybridMovieRecommender()

# Model file paths
model_filepath = 'my_recommender_model.h5'
user_encoder_filepath = 'user_encoder.pkl'
movie_encoder_filepath = 'movie_encoder.pkl'

# Try to load an existing model
if os.path.exists(model_filepath):
    logger.info("Loading existing model...")
    recommender.load_model(model_filepath)
    if os.path.exists(user_encoder_filepath):
        with open(user_encoder_filepath, 'rb') as f:
            recommender.user_encoder = pickle.load(f)
    if os.path.exists(movie_encoder_filepath):
        with open(movie_encoder_filepath, 'rb') as f:
            recommender.movie_encoder = pickle.load(f)
else:
    logger.info("No existing model found. Training a new model...")
    recommender.train_model(epochs=5)
    recommender.save_model(model_filepath)
    with open(user_encoder_filepath, 'wb') as f:
        pickle.dump(recommender.user_encoder, f)
    with open(movie_encoder_filepath, 'wb') as f:
        pickle.dump(recommender.movie_encoder, f)

recommender.schedule_retraining(frequency_days=7)  # Retrain weekly



@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    try:
        # Get the request data
        data = request.json
        
        # Extract swipe data if present
        swipes = data.pop('swipes', []) if isinstance(data, dict) else []
        
        # Use the remaining data as preferences
        preferences = data
        
        logger.info(f"Received preferences: {preferences}")
        logger.info(f"Received {len(swipes)} swipe records")
        
        # Create a truly unique ID for this session
        from uuid import uuid4
        temp_user_id = f"temp-{uuid4()}"
        
        # Clear any existing temporary users
        recommender.users = [u for u in recommender.users if not str(u['user_id']).startswith('temp-')]
        
        # Process swipe data for user profile
        watched_movies = []
        ratings = []
        
        for swipe in swipes:
            movie_id = swipe.get('movieId')
            liked = swipe.get('liked', False)
            if movie_id:
                watched_movies.append(movie_id)
                # Convert to rating: 5 for liked, 1 for disliked
                ratings.append(5 if liked else 1)
        
        # Create new user profile with both preferences and swipe data
        user_profile = {
            'user_id': temp_user_id,
            'preferences': preferences,
            'watched_movies': watched_movies,
            'ratings': ratings
        }
        
        # Add the new user
        recommender.users.append(user_profile)
        
        # If user skipped all movies, log this fact
        if not watched_movies:
            logger.info("User did not like/dislike any movies. Recommendations will be based on quiz preferences only.")
        
        # Generate recommendations
        logger.info("Generating recommendations...")
        recommendations = recommender.recommend(user_id=temp_user_id, top_n=10)
        
        # Handle empty recommendations
        if recommendations.empty:
            logger.warning("No recommendations found that match all criteria")
            return jsonify({
                "status": "success",
                "recommendations": [],
                "message": "No movies matched your criteria. Try different preferences."
            })
            
        # Add additional information to the response if available
        if 'Year' in recommender.movies.columns and not recommendations.empty:
            recommendations = recommendations.merge(
                recommender.movies[['MovieID', 'Title', 'Year', 'CleanTitle', 'Genres']], 
                on='MovieID', 
                how='left'
            )
        
        # Convert to a list of dictionaries for JSON response
        results = recommendations.to_dict('records')
        logger.info(f"Generated {len(results)} recommendations")
        
        return jsonify({
            "status": "success",
            "recommendations": results
        })
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "recommendations": []  # Always include empty recommendations array
        }), 500
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/swipe', methods=['POST'])
def record_swipe():
    try:
        swipe_data = request.json
        movie_id = swipe_data.get('movieId')
        liked = swipe_data.get('liked', False)
        
        logger.info(f"Recorded swipe for movie {movie_id}: {'LIKED' if liked else 'DISLIKED'}")
        
        return jsonify({
            "status": "success",
            "message": "Swipe recorded"
        })
    
    except Exception as e:
        logger.error(f"Error recording swipe: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Add to api.py

@app.route('/api/model/status', methods=['GET'])
def model_status():
    try:
        # Read model version info
        if os.path.exists('model_version.json'):
            with open('model_version.json', 'r') as f:
                version_info = json.load(f)
        else:
            version_info = {"current_version": 1, "versions": []}
        
        # Get basic model stats
        model_stats = {
            "users_count": len(recommender.users),
            "movies_count": len(recommender.movies),
            "ratings_count": len(recommender.ratings)
        }
        
        return jsonify({
            "status": "success",
            "model": {
                "version": version_info.get("current_version", 1),
                "last_trained": version_info.get("versions", [{}])[-1].get("timestamp"),
                "stats": model_stats
            }
        })
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
    # Add this endpoint to api.py

@app.route('/api/swipe-candidates', methods=['POST'])
def get_swipe_candidates():
    try:
        # Get preferences from quiz
        preferences = request.json
        logger.info(f"Fetching swipe candidates with preferences: {preferences}")
        
        # Generate a sample of movies for swiping from our dataset
        sample_size = 15  # Number of movies to swipe through
        
        # Apply basic filtering based on preferences
        filtered_movies = recommender.movies.copy()
        
        # Apply genre filtering if specified
        if preferences.get('genres_liked'):
            liked_genres = [g.lower() for g in preferences.get('genres_liked', [])]
            filtered_movies = filtered_movies[filtered_movies['Genres'].apply(
                lambda g: any(lg in g.lower() for lg in liked_genres) if isinstance(g, str) else False
            )]
            logger.info(f"Filtered to {len(filtered_movies)} movies after genre filter")
        
        # Exclude disliked genres
        if preferences.get('genres_disliked'):
            disliked_genres = [g.lower() for g in preferences.get('genres_disliked', [])]
            filtered_movies = filtered_movies[~filtered_movies['Genres'].apply(
                lambda g: any(dg in g.lower() for dg in disliked_genres) if isinstance(g, str) else False
            )]
            logger.info(f"Filtered to {len(filtered_movies)} movies after excluding disliked genres")
        
        # Apply time period filter if specified
        if preferences.get('time_period') == 'Modern (1990-present)' and 'Year' in filtered_movies.columns:
            filtered_movies = filtered_movies[filtered_movies['Year'] >= 1990]
            logger.info(f"Filtered to {len(filtered_movies)} modern movies")
        elif preferences.get('time_period') == 'Classic (pre-1990)' and 'Year' in filtered_movies.columns:
            filtered_movies = filtered_movies[filtered_movies['Year'] < 1990]
            logger.info(f"Filtered to {len(filtered_movies)} classic movies")
        
        # If too few movies match criteria, expand selection
        if len(filtered_movies) < sample_size:
            logger.info("Too few movies match criteria for swiping. Expanding selection.")
            # Try with just genre filter
            if preferences.get('genres_liked'):
                primary_genre = preferences['genres_liked'][0].lower()
                filtered_movies = recommender.movies[recommender.movies['Genres'].apply(
                    lambda g: primary_genre in g.lower() if isinstance(g, str) else False
                )]
                logger.info(f"Expanded to {len(filtered_movies)} movies using primary genre filter")
            
            # If still too few, use random selection
            if len(filtered_movies) < sample_size:
                filtered_movies = recommender.movies.sample(min(sample_size * 2, len(recommender.movies)))
                logger.info(f"Using random selection of {len(filtered_movies)} movies")
        
        # Sample movies and prepare response
        swipe_candidates = filtered_movies.sample(min(sample_size, len(filtered_movies)))
        
        # Create response with necessary fields
        results = []
        for _, row in swipe_candidates.iterrows():
            movie_data = {
                'MovieID': int(row['MovieID']),
                'Title': row['Title'],
                'Genres': row['Genres']
            }
            # Add Year if available
            if 'Year' in row:
                movie_data['Year'] = int(row['Year'])
            # Add CleanTitle if available
            if 'CleanTitle' in row:
                movie_data['CleanTitle'] = row['CleanTitle']
            
            results.append(movie_data)
        
        logger.info(f"Returning {len(results)} swipe candidates")
        return jsonify({
            "status": "success",
            "candidates": results
        })
    except Exception as e:
        logger.error(f"Error getting swipe candidates: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)