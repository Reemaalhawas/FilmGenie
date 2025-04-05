# recommendations.py
from flask import Blueprint, jsonify
from flask_login import login_required, current_user
from models import User
from ncf_model import MovieRecommender
from questionnaire import MovieQuestionnaire
import pandas as pd

reco_bp = Blueprint('reco', __name__)

# Initialize or load the recommendation model (could be loaded from a saved file in a real scenario)
# Here, we'll train it on startup for simplicity (this may take time if dataset is large).
recommender = MovieRecommender()
# Load datasets
ratings_df = pd.read_csv('data/ratings_updated.csv')
movies_df = pd.read_csv('data/movies_with_actors.csv')   # movie metadata, if needed
# Preprocess data and train model 
# (In practice, you might load a pre-trained model to avoid long startup time)
try:
    X_user, X_movie, X_features, y = recommender.preprocess_data(
        ratings_df, movies_df, pd.DataFrame()  # no specific user pref DataFrame for training
    )
    recommender.train(X_user, X_movie, X_features, y, epochs=5)  # train with fewer epochs for demo
except Exception as e:
    print("Error training recommendation model:", e)

# Helper: format recommendations with movie details
def format_recommendations(rec_list):
    """Convert list of (movie_id, score) to movie info dicts for frontend."""
    if rec_list is None:
        return []
    # Load movies DataFrame if not loaded
    global movies_df
    if movies_df is None:
        movies_df = pd.read_csv('data/movies_with_actors.csv')
    movies_map = { int(mid): {"id": int(mid), "title": row.Title, "genres": row.Genres, "actors": row.Actors}
                   for mid, row in movies_df.set_index("MovieID").iterrows() }
    formatted = []
    for mid, score in rec_list:
        if int(mid) in movies_map:
            info = movies_map[int(mid)]
            info["score"] = float(score)
            formatted.append(info)
        else:
            formatted.append({"id": int(mid), "score": float(score), "title": "Movie "+str(mid)})
    return formatted

@reco_bp.route("/api/recommendations")
@login_required
def get_recommendations():
    """API endpoint to get personalized movie recommendations for the logged-in user."""
    user: User = current_user  # type: ignore
    # If user has no recorded preferences, just return some popular movies as fallback
    if user.get_preferences() is None:
        # Could either prompt for onboarding or return generic top movies
        return jsonify({"needs_onboarding": True, "recommendations": []})
    # If we have preferences, use them to generate recommendations
    prefs_dict = user.get_preferences()
    # Convert preferences dict to feature vector using MovieQuestionnaire
    questionnaire = MovieQuestionnaire()
    user_pref_vector = questionnaire.preprocess_preferences(prefs_dict)
    # Use a dummy user_id for prediction (since this user wasn’t in training set)
    # We'll use a special user_id = 0 (if the model was trained with users labeled 1..N, 0 might be new)
    try:
        recs = recommender.get_recommendations(user_id=0, user_preferences=user_pref_vector, top_n=10)
    except Exception as e:
        # If model isn't trained or user 0 not in encoder, handle gracefully
        return jsonify({"error": f"Recommendation model error: {e}"}), 500

    formatted_recs = format_recommendations(recs)
    return jsonify({"needs_onboarding": False, "recommendations": formatted_recs})

@reco_bp.route("/api/preferences", methods=["POST"])
@login_required
def save_preferences():
    """API endpoint to save onboarding questionnaire responses for the current user."""
    from flask import request
    user: User = current_user  # type: ignore
    data = request.get_json()
    if not data:
        return {"error": "No data provided"}, 400
    # Here, `data` is expected to be a dict of {question_key: answer_value}, 
    # e.g., {"mood": "Happy", "genres_liked": ["Action","Comedy"], ...}
    # Validate and save preferences
    user.set_preferences(data)
    db.session.commit()
    return {"message": "Preferences saved successfully"}, 200
