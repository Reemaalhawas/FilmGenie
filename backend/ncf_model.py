import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.metrics import Precision, Recall
import threading
import time
from datetime import datetime, timedelta


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BACKEND_DIR, 'data')

class HybridMovieRecommender:
    def __init__(self):
        self.movies = self._load_movies()
        self.ratings = self._load_ratings()
        self.users = self._load_fake_users()
        self.genre_list = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama']
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.model = None

    def _load_movies(self):
        path = os.path.join(DATA_DIR, 'movies_with_actors_and_keywords.csv')
        df = pd.read_csv(path)
        
        df['genres'] = df['Genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
        df['keywords'] = df['Keywords'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
        return df

    def _load_ratings(self):
        path = os.path.join(DATA_DIR, 'ratings_updated.csv')
        return pd.read_csv(path)

    def _load_fake_users(self):
        path = os.path.join(DATA_DIR, 'fake_user_profiles.json')
        try:
            with open(path, 'r') as file:
                return json.load(file)['users']
        except FileNotFoundError:
            logger.warning(f"Fake user profiles file not found at {path}. Initializing with empty list.")
            return []
        except json.JSONDecodeError:
            logger.warning(f"Error decoding fake user profiles file at {path}. Initializing with empty list.")
            return []

    def _genre_vector(self, genres):
        return np.array([1 if genre in genres else 0 for genre in self.genre_list], dtype=np.float32)

    def _preference_vector(self, prefs):
        
        pref_vector = np.zeros(60, dtype=np.float32)
        
        
        mood_map = {
            'Happy': 0, 'Sad': 1, 'Excited': 2, 'Relaxed': 3, 
            'Thoughtful': 4, 'Energetic': 5, 'Stressed': 6
        }
        if prefs.get('mood') in mood_map:
            pref_vector[mood_map[prefs.get('mood')]] = 1.0
        
        
        desired_mood_map = {
            'Happy': 7, 'Inspired': 8, 'Excited': 9, 'Thoughtful': 10, 
            'Relaxed': 11, 'Thrilled': 12
        }
        if prefs.get('desired_mood') in desired_mood_map:
            pref_vector[desired_mood_map[prefs.get('desired_mood')]] = 1.0
        
        
        attention_map = {
            'Casual (can multitask)': 13, 
            'Moderate (some focus needed)': 14, 
            'Full attention required': 15
        }
        if prefs.get('attention_level') in attention_map:
            pref_vector[attention_map[prefs.get('attention_level')]] = 1.0
        
        
        complexity_map = {
            'Prefer simple plots': 16, 
            'Like some complexity': 17, 
            'Love complex mind-bending stories': 18
        }
        if prefs.get('plot_complexity') in complexity_map:
            pref_vector[complexity_map[prefs.get('plot_complexity')]] = 1.0
        
        
        pacing_map = {
            'Slow and steady': 19, 
            'Moderate pace': 20, 
            'Fast-paced': 21
        }
        if prefs.get('pacing') in pacing_map:
            pref_vector[pacing_map[prefs.get('pacing')]] = 1.0
        
       
        length_map = {
            'Prefer shorter movies': 22, 
            "Don't mind longer movies": 23, 
            'Love epic length movies': 24
        }
        if prefs.get('movie_length') in length_map:
            pref_vector[length_map[prefs.get('movie_length')]] = 1.0
        
       
        language_map = {
            'English': 25, 
            'Foreign language': 26, 
            'No preference': 27
        }
        if prefs.get('language') in language_map:
            pref_vector[language_map[prefs.get('language')]] = 1.0
        
        
        period_map = {
            'Classic (pre-1990)': 28, 
            'Modern (1990-present)': 29, 
            'No preference': 30
        }
        if prefs.get('time_period') in period_map:
            pref_vector[period_map[prefs.get('time_period')]] = 1.0
        
        
        rating_map = {
            'Yes, highly rated only': 31, 
            'No preference': 32
        }
        if prefs.get('rating') in rating_map:
            pref_vector[rating_map[prefs.get('rating')]] = 1.0
        
       
        genre_list = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama']
        if prefs.get('genres_liked'):
            for genre in prefs.get('genres_liked'):
                if genre in genre_list:
                    idx = 33 + genre_list.index(genre)
                    pref_vector[idx] = 1.0
        
        
        if prefs.get('genres_disliked'):
            for genre in prefs.get('genres_disliked'):
                if genre in genre_list:
                    idx = 41 + genre_list.index(genre)
                    pref_vector[idx] = 1.0
        
        
        platform_map = {
            'Netflix': 49, 
            'Amazon Prime': 50, 
            'Hulu': 51, 
            'Disney+': 52, 
            'Other': 53
        }
        if isinstance(prefs.get('streaming_platform'), list):
            for platform in prefs.get('streaming_platform'):
                if platform in platform_map:
                    pref_vector[platform_map[platform]] = 1.0
        elif prefs.get('streaming_platform') in platform_map:
            pref_vector[platform_map[prefs.get('streaming_platform')]] = 1.0
        
        
        if prefs.get('genres_liked'):
            pref_vector[54] = min(len(prefs.get('genres_liked')) / 5.0, 1.0)  # Normalize to [0,1]
        
        
        if prefs.get('genres_disliked'):
            pref_vector[55] = min(len(prefs.get('genres_disliked')) / 5.0, 1.0)  # Normalize to [0,1]
        
        
        rating_value_map = {'G': 0.0, 'PG': 0.25, 'PG-13': 0.5, 'R': 0.75, 'Any': 1.0}
        if prefs.get('rating') in rating_value_map:
            pref_vector[56] = rating_value_map[prefs.get('rating')]
        
        
        novelty_score = 0.0
        if prefs.get('plot_complexity') == 'Love complex mind-bending stories':
            novelty_score += 0.5
        if prefs.get('language') == 'Foreign language':
            novelty_score += 0.3
        if prefs.get('time_period') == 'Classic (pre-1990)':
            novelty_score += 0.2
        pref_vector[57] = min(novelty_score, 1.0)
        
        
        
        return pref_vector

    def _build_model(self):
        num_users = len(self.ratings['UserID'].unique())
        num_movies = len(self.ratings['MovieID'].unique())

        user_input = Input(shape=(1,), name='user_input')
        user_embed = Flatten()(Embedding(num_users + 1, 50)(user_input))

        movie_input = Input(shape=(1,), name='movie_input')
        movie_embed = Flatten()(Embedding(num_movies + 1, 50)(movie_input))

        features_input = Input(shape=(68,), name='features_input')
        features_dense = Dense(64, activation='relu')(features_input)

        x = Concatenate()([user_embed, movie_embed, features_dense])
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)

        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[user_input, movie_input, features_input], outputs=output)
       
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
        self.model = model

    def train_model(self, epochs=5, batch_size=128):
        logger.info("Preparing training data...")

        
        self.user_encoder.fit(self.ratings['UserID'])
        self.movie_encoder.fit(self.ratings['MovieID'])

        
        X_user = self.user_encoder.transform(self.ratings['UserID']).reshape(-1, 1)
        X_movie = self.movie_encoder.transform(self.ratings['MovieID']).reshape(-1, 1)
        y = (self.ratings['Rating'] >= 4).astype(int).values

       
        movie_genre_map = {
            row['MovieID']: self._genre_vector(row['genres'])
            for _, row in self.movies.iterrows()
        }

        features = []
        for _, row in self.ratings.iterrows():
            user_prefs = next((u['preferences'] for u in self.users if u['user_id'] == row['UserID']), None)
            if user_prefs:
                pref_vec = self._preference_vector(user_prefs)
                genre_vec = movie_genre_map.get(row['MovieID'], np.zeros(len(self.genre_list)))
                features.append(np.concatenate([pref_vec, genre_vec]))
            else:
                features.append(np.concatenate([np.zeros(60), np.zeros(len(self.genre_list))]))
        X_features = np.array(features)

        self._build_model()
        logger.info("Training model...")
        history = self.model.fit(
            [X_user, X_movie, X_features], y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
       
        self.model.history = history
        logger.info("Training complete.")

    def recommend(self, user_id, top_n=10):
        logger.info(f"Generating recommendations for user {user_id}...")

        target_user = next((u for u in self.users if u['user_id'] == user_id), None)
        if not target_user:
            logger.warning("User not found.")
            return pd.DataFrame(columns=['MovieID', 'Title'])

        target_vector = self._preference_vector(target_user['preferences'])
        prefs = target_user['preferences']

       
        logger.info(f"User preferences: {prefs}")

        
        filtered_movies = self.movies.copy()

        
        if prefs.get('genres_disliked'):
            logger.info(f"Filtering out disliked genres: {prefs['genres_disliked']}")
            filtered_movies = filtered_movies[~filtered_movies['genres'].apply(
                lambda g: any(d.lower() in [x.lower() for x in g] for d in prefs['genres_disliked'])
            )]

        
        if prefs.get('genres_liked'):
            logger.info(f"Filtering for liked genres: {prefs['genres_liked']}")
            
            liked_genres = [g.lower() for g in prefs['genres_liked']]
            filtered_movies = filtered_movies[filtered_movies['genres'].apply(
                lambda g: any(liked_genre in [x.lower() for x in g] for liked_genre in liked_genres)
            )]

        
        if prefs.get('time_period_preference') == "Modern (1990-present)":
            filtered_movies = filtered_movies[filtered_movies['Title'].str.extract(r'\((\d{4})\)').astype(float) >= 1990]
        elif prefs.get('time_period_preference') == "Classic (pre-1990)":
            filtered_movies = filtered_movies[filtered_movies['Title'].str.extract(r'\((\d{4})\)').astype(float) < 1990]

        
        mood_keyword_map = {
            'Happy': ['fun', 'joyful', 'uplifting', 'lighthearted', 'inspirational', 'comedy', 'humor', 'happy', 'feel-good'],
            'Sad': ['tragic', 'heartbreak', 'loss', 'emotional', 'drama', 'sad'],
            'Excited': ['thrilling', 'action', 'intense', 'adventure', 'exciting'],
            'Relaxed': ['calm', 'romantic', 'slow', 'soothing', 'peaceful'],
            'Thoughtful': ['philosophical', 'complex', 'deep', 'thought-provoking', 'intellectual']
        }
        desired_mood = prefs.get('desired_mood', '')
        if desired_mood in mood_keyword_map:
            mood_keywords = mood_keyword_map[desired_mood]
            filtered_movies = filtered_movies[filtered_movies['keywords'].apply(
                lambda k: sum(1 for m in mood_keywords if m in ' '.join(k).lower()) >= 2  # Require at least 2 keyword matches
            )]

        
        if prefs.get('plot_complexity') == 'simple':
            filtered_movies = filtered_movies[filtered_movies['keywords'].apply(
                lambda k: not any(word in ' '.join(k).lower() for word in ['complex', 'complicated', 'intricate', 'dark', 'serious', 'intense'])
            )]
        elif prefs.get('plot_complexity') == 'complex':
            filtered_movies = filtered_movies[filtered_movies['keywords'].apply(
                lambda k: any(word in ' '.join(k).lower() for word in ['complex', 'complicated', 'intricate'])
            )]

        if prefs.get('pace') == 'slow':
            filtered_movies = filtered_movies[filtered_movies['keywords'].apply(
                lambda k: any(word in ' '.join(k).lower() for word in ['slow', 'leisurely', 'relaxed', 'gentle', 'easygoing'])
            )]
        elif prefs.get('pace') == 'fast':
            filtered_movies = filtered_movies[filtered_movies['keywords'].apply(
                lambda k: any(word in ' '.join(k).lower() for word in ['fast', 'action', 'thrilling'])
            )]

        
        filtered_movies = filtered_movies[filtered_movies['keywords'].apply(
            lambda k: not any(word in ' '.join(k).lower() for word in ['war', 'crime', 'horror', 'thriller', 'violence', 'dark'])
        )]

        
        logger.info(f"Number of movies after all filtering: {len(filtered_movies)}")
        
        
        if len(filtered_movies) == 0:
            logger.warning("No movies match all criteria, relaxing filters...")
            
            filtered_movies = self.movies.copy()
            
            
            if prefs.get('genres_disliked'):
                filtered_movies = filtered_movies[~filtered_movies['genres'].apply(
                    lambda g: any(d.lower() in [x.lower() for x in g] for d in prefs['genres_disliked'])
                )]
                
            if prefs.get('genres_liked'):
                liked_genres = [g.lower() for g in prefs['genres_liked']]
                filtered_movies = filtered_movies[filtered_movies['genres'].apply(
                    lambda g: any(liked_genre in [x.lower() for x in g] for liked_genre in liked_genres)
                )]
                
            logger.info(f"After relaxed filtering: {len(filtered_movies)} movies")
            
            
            if len(filtered_movies) == 0:
                logger.warning("Still no matches, returning a selection of movies")
                filtered_movies = self.movies.sample(min(30, len(self.movies)))

        similarities = []
        for user in self.users:
            if user['user_id'] == user_id:
                continue
            vec = self._preference_vector(user['preferences'])
            sim = cosine_similarity([target_vector], [vec])[0][0]
            similarities.append((user, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_users = similarities[:5]

        recommended = {}
        for user, score in similar_users:
            for movie_id, rating in zip(user['watched_movies'], user['ratings']):
                if rating >= 4 and movie_id in filtered_movies['MovieID'].values:
                    recommended[movie_id] = recommended.get(movie_id, 0) + score

        sorted_recs = sorted(recommended.items(), key=lambda x: x[1], reverse=True)
        movie_ids = [mid for mid, _ in sorted_recs 
                    if mid not in target_user['watched_movies'] 
                    and mid in filtered_movies['MovieID'].values]

        
        if len(movie_ids) < top_n:
            
            remaining_movies = filtered_movies[
                ~filtered_movies['MovieID'].isin(movie_ids) & 
                ~filtered_movies['MovieID'].isin(target_user['watched_movies'])
            ]
           
            remaining_movies = remaining_movies.sort_values('Title')
            additional_ids = remaining_movies['MovieID'].values[:top_n - len(movie_ids)]
            movie_ids.extend(additional_ids)

        
        if len(movie_ids) == 0:
            logger.warning("No movies to recommend after all processing")
            return pd.DataFrame(columns=['MovieID', 'Title'])

        movie_ids = movie_ids[:top_n]

       
        valid_movie_ids = [mid for mid in movie_ids if mid in self.movie_encoder.classes_]
        
        
        if not valid_movie_ids:
            logger.warning("No valid movie IDs found in recommendation set")
            
            encoder_movies = [mid for mid in self.movie_encoder.classes_ if mid in self.movies['MovieID'].values]
            valid_movie_ids = encoder_movies[:min(top_n, len(encoder_movies))]
            
            
            if not valid_movie_ids:
                logger.error("Cannot find any valid movies for recommendation")
                return pd.DataFrame(columns=['MovieID', 'Title'])
        
       
        movie_ids = valid_movie_ids
        movie_inputs = self.movie_encoder.transform(movie_ids).reshape(-1, 1)
        
        
        if user_id not in self.user_encoder.classes_:
           
            logger.warning(f"User {user_id} not in encoder, using a substitute user")
            substitute_user_id = self.user_encoder.classes_[0]  # Use the first known user as substitute
            user_inputs = np.full(len(movie_ids), self.user_encoder.transform([substitute_user_id])[0]).reshape(-1, 1)
        else:
            user_inputs = np.full(len(movie_ids), self.user_encoder.transform([user_id])[0]).reshape(-1, 1)

        movie_genre_map = {
            row['MovieID']: self._genre_vector(row['genres'])
            for _, row in self.movies.iterrows()
        }
        
        features = []
        for movie_id in movie_ids:
            genre_vec = movie_genre_map.get(movie_id, np.zeros(len(self.genre_list)))
            features.append(np.concatenate([target_vector, genre_vec]))
        features = np.array(features)

        
        if len(movie_ids) == 0:
            logger.warning("No movies available for prediction")
            return pd.DataFrame(columns=['MovieID', 'Title'])

        try:
            predictions = self.model.predict([user_inputs, movie_inputs, features])
            results = list(zip(movie_ids, predictions.flatten()))
            results.sort(key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            
            results = [(mid, 0.5) for mid in movie_ids]
        
        
        recommended_movie_ids = [r[0] for r in results]
        recommended_movies = self.movies[self.movies['MovieID'].isin(recommended_movie_ids)]
        
         
        if recommended_movies.empty:
            logger.warning("No recommended movies found in dataset")
            return pd.DataFrame(columns=['MovieID', 'Title'])
        
        
        logger.info("\nFinal Recommendations:")
        for _, row in recommended_movies.iterrows():
            logger.info(f"Movie: {row['Title']}, Genres: {row['genres']}")
            
        return recommended_movies[['MovieID', 'Title']].reset_index(drop=True)

    def evaluate_model(self):
        logger.info("Evaluating model...")
        if not self.model or not hasattr(self.model, 'history') or not self.model.history.history:
            logger.warning("No training history available. Consider evaluating on a test set using model.evaluate().")
            return None

        history = self.model.history.history
        
        logger.info("History keys: " + str(list(history.keys())))

        
        metrics = {
            'loss': history.get('loss', [])[-1] if 'loss' in history and history['loss'] else None,
            'val_loss': history.get('val_loss', [])[-1] if 'val_loss' in history and history['val_loss'] else None,
            'accuracy': history.get('accuracy', [])[-1] if 'accuracy' in history and history['accuracy'] else None,
            'val_accuracy': history.get('val_accuracy', [])[-1] if 'val_accuracy' in history and history['val_accuracy'] else None,
            'precision': history.get('precision', [])[-1] if 'precision' in history and history['precision'] else None,
            'val_precision': history.get('val_precision', [])[-1] if 'val_precision' in history and history['val_precision'] else None,
            'recall': history.get('recall', [])[-1] if 'recall' in history and history['recall'] else None,
            'val_recall': history.get('val_recall', [])[-1] if 'val_recall' in history and history['val_recall'] else None
        }

        logger.info("Final model performance (loss, accuracy, precision, recall):")
        for key, value in metrics.items():
            if value is not None:
                logger.info(f"{key}: {value:.4f}")
        return metrics

    
    def save_model(self, filepath='my_recommender_model.h5'):
        if self.model:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        else:
            logger.warning("No model to save.")

    def load_model(self, filepath='my_recommender_model.h5'):
        try:
            self.model = load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

    

    def schedule_retraining(self, frequency_days=7):
        """Sets up periodic model retraining"""
        
        def retraining_job():
            while True:
                
                next_training = datetime.now() + timedelta(days=frequency_days)
                logger.info(f"Next model retraining scheduled for: {next_training}")
                
                
                sleep_seconds = (next_training - datetime.now()).total_seconds()
                time.sleep(sleep_seconds)
                
                try:
                    logger.info("Starting scheduled model retraining...")
                    
                    
                    import shutil
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = f"model_backup_{timestamp}.h5"
                    shutil.copy2('my_recommender_model.h5', backup_path)
                    
                    
                    baseline_metrics = self.evaluate_model()
                    
                    
                    self.train_model(epochs=5)
                    
                    
                    new_metrics = self.evaluate_model()
                    
                    
                    improvements = {}
                    for metric, value in new_metrics.items():
                        if baseline_metrics.get(metric) is not None:
                            change = value - baseline_metrics[metric]
                            improvements[metric] = change
                    
                    
                    from model_versioning import update_model_version
                    new_version = update_model_version({
                        "baseline": baseline_metrics,
                        "new": new_metrics,
                        "improvements": improvements
                    })
                    
                    logger.info(f"Model updated to version {new_version}")
                    
                    
                    self.save_model('my_recommender_model.h5')
                        
                   
                    
                except Exception as e:
                    logger.error(f"Error during scheduled retraining: {str(e)}")
                    
                    try:
                        shutil.copy2(backup_path, 'my_recommender_model.h5')
                        logger.info("Restored previous model from backup")
                    except Exception as restore_error:
                        logger.error(f"Failed to restore model: {str(restore_error)}")

        
        retraining_thread = threading.Thread(target=retraining_job, daemon=True)
        retraining_thread.start()
        logger.info("Scheduled periodic model retraining")

        

    def update_model_version(performance_metrics=None):
        """Update the model version information file"""
        version_file = 'model_version.json'
        
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                version_info = json.load(f)
        else:
            version_info = {
                "versions": []
            }
        
       
        new_version = {
            "version": len(version_info["versions"]) + 1,
            "timestamp": datetime.now().isoformat(),
            "metrics": performance_metrics or {}
        }
        
        
        version_info["versions"].append(new_version)
        version_info["current_version"] = new_version["version"]
        
        
        with open(version_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        return new_version["version"]
