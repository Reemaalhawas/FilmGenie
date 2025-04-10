import os
import json
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.layers import (
    Input, Dense, Embedding, Flatten, Concatenate,
    Dropout, BatchNormalization, Activation
)
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import AUC, Precision, Recall
from keras.initializers import GlorotNormal
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine project root (for example purposes)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class MovieRecommender:
    def __init__(self, movies_path=None, ratings_path=None):
        """
        Initialize the MovieRecommender with paths to data files.
        
        Args:
            movies_path (str): Path to the movies.dat file.
            ratings_path (str): Path to the ratings CSV file.
        """
        self.logger = logging.getLogger(__name__)
        self.movies = None
        self.ratings = None
        self.model = None
        self.training_history = None
        
        # List of genres to be used in profile vectors.
        self.genre_list = [
            'Action', 'Adventure', 'Animation', 'Children',
            'Comedy', 'Crime', 'Documentary', 'Drama'
        ]
        self.movie_to_idx = {}
        self.user_ratings = {}  # Dictionary to store user ratings
        
        # Encoders for user and movie IDs.
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        
        # Optionally load movies and ratings data.
        if movies_path:
            self.load_movies(movies_path)
        if ratings_path:
            self.load_ratings(ratings_path)

    def load_movies(self, movies_path):
        """
        Load movies data from a file and store as a DataFrame.
        
        Each line in the file is assumed to be in the format:
            movieId::title::genres
        
        Args:
            movies_path (str): Path to the movies.dat file.
        """
        try:
            movies_data = []
            with open(movies_path, 'r', encoding='utf-8') as f:
                for line in f:
                    movie_id, title, genres = line.strip().split('::')
                    genres_list = genres.split('|')
                    movies_data.append({
                        'movieId': int(movie_id),
                        'title': title,
                        'genres': genres_list
                    })
            
            movies_df = pd.DataFrame(movies_data)
            self.movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies_df['movieId'])}
            self.movies = movies_df
            self.logger.info(f"Loaded {len(movies_df)} movies from {movies_path}")
        except Exception as e:
            self.logger.error(f"Error loading movies data: {str(e)}")
            raise

    def load_ratings(self, ratings_path):
        """
        Load ratings from a CSV file and encode the IDs.
        
        Args:
            ratings_path (str): Path to the ratings CSV file.
        """
        try:
            if not os.path.exists(ratings_path):
                self.logger.warning(f"Ratings file not found: {ratings_path}")
                self.ratings = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
                return
            
            self.ratings = pd.read_csv(ratings_path)
            # Check required columns.
            required_columns = ['userId', 'movieId', 'rating']
            missing_columns = [col for col in required_columns if col not in self.ratings.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in ratings: {missing_columns}")
            
            # Encode user and movie IDs.
            self.user_encoder.fit(self.ratings['userId'].unique())
            self.movie_encoder.fit(self.ratings['movieId'].unique())
            self.logger.info(f"Loaded {len(self.ratings)} ratings")
            self.logger.info(f"Number of unique users: {len(self.user_encoder.classes_)}")
            self.logger.info(f"Number of unique movies: {len(self.movie_encoder.classes_)}")
        except Exception as e:
            self.logger.error(f"Error loading ratings: {str(e)}")
            self.ratings = pd.DataFrame(columns=['userId', 'movieId', 'rating'])

    def build_ncf_model(self, num_users, num_movies, embedding_dim=50):
        """
        Build the neural collaborative filtering model.
        
        Args:
            num_users (int): Number of unique users.
            num_movies (int): Number of unique movies.
            embedding_dim (int): Dimensionality for embedding layers.
        
        Returns:
            keras.Model: The compiled model.
        """
        # User input and embedding
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(
            input_dim=num_users + 1,  # +1 for padding
            output_dim=embedding_dim,
            name='user_embedding'
        )(user_input)
        user_embedding = Flatten()(user_embedding)

        # Movie input and embedding
        movie_input = Input(shape=(1,), name='movie_input')
        movie_embedding = Embedding(
            input_dim=len(self.movie_to_idx) + 1,  # +1 for padding
            output_dim=embedding_dim,
            name='movie_embedding'
        )(movie_input)
        movie_embedding = Flatten()(movie_embedding)

        # Feature input (68 features in total)
        feature_input = Input(shape=(68,), name='feature_input')
        feature_dense = Dense(64, activation='relu')(feature_input)

        # Concatenate user, movie, and feature representations.
        concat = Concatenate()([user_embedding, movie_embedding, feature_dense])

        # MLP layers with BatchNormalization and Dropout.
        x = Dense(128, activation='relu')(concat)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)

        # Output layer for binary classification.
        output = Dense(1, activation='sigmoid')(x)

        model = Model(
            inputs=[user_input, movie_input, feature_input],
            outputs=output
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', AUC(), Precision(), Recall()]
        )

        self.model = model
        self.logger.info("Neural Collaborative Filtering model built successfully")
        return model

    def _process_preferences(self, preferences):
        """
        Convert user preferences into a 60-dimensional feature vector.
        
        The vector is constructed from various categories:
         - Mood, desired feeling: 5+5 features.
         - Liked & disliked genres: 10+10 features.
         - Attention, plot complexity, pacing, movie length, language: 3 features each.
         - Time period, rating preference, streaming platform: 5, 5, and 5 features respectively.
        
        Args:
            preferences (dict): User preferences.
        
        Returns:
            np.array: 60-dimensional feature vector.
        """
        try:
            feature_vector = np.zeros(60)
            idx = 0

            # Mood (5 features)
            moods = ['Happy', 'Sad', 'Excited', 'Relaxed', 'Thoughtful']
            mood = preferences.get('mood', '')
            if mood in moods:
                feature_vector[moods.index(mood)] = 1
            idx += 5

            # Desired feeling (5 features)
            feelings = ['Happy', 'Sad', 'Excited', 'Relaxed', 'Thoughtful']
            feeling = preferences.get('desired_feeling', '')
            if feeling in feelings:
                feature_vector[idx + feelings.index(feeling)] = 1
            idx += 5

            # Liked genres (10 features)
            genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance',
                      'Sci-Fi', 'Thriller', 'Documentary', 'Animation', 'Fantasy']
            liked_genres = preferences.get('liked_genres', [])
            for genre in liked_genres:
                if genre in genres:
                    feature_vector[idx + genres.index(genre)] = 1
            idx += 10

            # Disliked genres (10 features)
            disliked_genres = preferences.get('disliked_genres', [])
            for genre in disliked_genres:
                if genre in genres:
                    feature_vector[idx + genres.index(genre)] = 1
            idx += 10

            # Attention level (3 features)
            attention_levels = ['Casual', 'Moderate', 'Full']
            attention = preferences.get('attention_level', '')
            if attention in attention_levels:
                feature_vector[idx + attention_levels.index(attention)] = 1
            idx += 3

            # Plot complexity (3 features)
            complexities = ['Simple', 'Moderate', 'Complex']
            complexity = preferences.get('plot_complexity', '')
            if complexity in complexities:
                feature_vector[idx + complexities.index(complexity)] = 1
            idx += 3

            # Pacing (3 features)
            pacing_options = ['Slow', 'Medium', 'Fast']
            pacing = preferences.get('pacing', '')
            if pacing in pacing_options:
                feature_vector[idx + pacing_options.index(pacing)] = 1
            idx += 3

            # Movie length (3 features)
            lengths = ['Short', 'Medium', 'Long']
            length = preferences.get('movie_length', '')
            if length in lengths:
                feature_vector[idx + lengths.index(length)] = 1
            idx += 3

            # Language (3 features)
            languages = ['English', 'Foreign', 'Both']
            language = preferences.get('language', '')
            if language in languages:
                feature_vector[idx + languages.index(language)] = 1
            idx += 3

            # Time period (5 features)
            periods = ['Classic', 'Modern', 'Contemporary', 'Any', 'Specific']
            period = preferences.get('time_period', '')
            if period in periods:
                feature_vector[idx + periods.index(period)] = 1
            idx += 5

            # Rating preference (5 features)
            ratings = ['G', 'PG', 'PG-13', 'R', 'Any']
            rating = preferences.get('rating', '')
            if rating in ratings:
                feature_vector[idx + ratings.index(rating)] = 1
            idx += 5

            # Streaming platform (5 features)
            platforms = ['Netflix', 'Amazon', 'Hulu', 'Disney+', 'Other']
            platform = preferences.get('streaming_platform', '')
            if platform in platforms:
                feature_vector[idx + platforms.index(platform)] = 1
            idx += 5

            assert idx == 60, f"Feature vector size mismatch: {idx} != 60"
            return feature_vector

        except Exception as e:
            self.logger.error(f"Error processing preferences: {str(e)}")
            raise

    def create_user_profile(self, user_data):
        """
        Create a user profile by combining a genre profile from the movies the user rated high
        and the processed preference vector.
        
        Args:
            user_data (dict): Dictionary containing user data.
        
        Returns:
            np.array: Combined profile vector.
        """
        try:
            vectors = []
            for movie_id, rating in zip(user_data.get('watched_movies', []), user_data.get('ratings', [])):
                if rating >= 4:
                    # Locate movie row by movieId.
                    row = self.movies[self.movies['movieId'] == movie_id]
                    if not row.empty:
                        genres = row.iloc[0]['genres']
                        vector = [1 if genre in genres else 0 for genre in self.genre_list]
                        vectors.append(vector)
            if vectors:
                profile_vector = np.mean(vectors, axis=0)
            else:
                profile_vector = np.zeros(len(self.genre_list))
            
            preferences_vector = self._process_preferences(user_data['preferences'])
            final_vector = np.concatenate([profile_vector, preferences_vector])
            self.logger.info("User profile created successfully")
            return final_vector

        except Exception as e:
            self.logger.error(f"Error creating user profile: {str(e)}")
            raise

    def preprocess_data(self, ratings_df, movies_df, user_preferences):
        """
        Preprocess training data by constructing input features and labels.
        Input features have 68 dimensions (60 from preferences, 8 from movie genre profile).
        
        Args:
            ratings_df (DataFrame): DataFrame containing rating data.
            movies_df (DataFrame): DataFrame containing movies data.
            user_preferences (dict or Series): User preference data.
        
        Returns:
            Tuple: (X_user, X_movie, X_features, y)
        """
        try:
            self.movies = movies_df
            self.movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies_df['movieId'])}
            ratings_df = ratings_df.rename(columns={'UserID': 'userId', 'MovieID': 'movieId', 'Rating': 'rating'})
            if hasattr(user_preferences, 'to_dict'):
                user_preferences = user_preferences.to_dict()

            X_user = self.user_encoder.fit_transform(ratings_df['userId'])
            X_movie = self.movie_encoder.fit_transform(ratings_df['movieId'])
            num_ratings = len(ratings_df)
            X_features = np.zeros((num_ratings, 68))

            # Map each movie to its 8-dimensional genre vector.
            movie_genre_map = {}
            for _, movie in movies_df.iterrows():
                genre_vector = np.zeros(8)
                for j, genre in enumerate(self.genre_list):
                    if genre in movie['genres']:
                        genre_vector[j] = 1
                movie_genre_map[movie['movieId']] = genre_vector

            movie_genres = np.array([movie_genre_map[movie_id] for movie_id in ratings_df['movieId']])
            X_features[:, 60:] = movie_genres

            # Set user preference features.
            liked_genres = user_preferences.get('genres_liked', [])
            if isinstance(liked_genres, str):
                liked_genres = [liked_genres]
            for i, genre in enumerate(self.genre_list):
                if genre in liked_genres:
                    X_features[:, i] = 1

            mood_idx = {'Happy': 0, 'Sad': 1, 'Energetic': 2, 'Thoughtful': 3, 'Excited': 4, 'Calm': 5, 'Stressed': 6}
            mood = str(user_preferences.get('mood', 'Happy'))
            if mood in mood_idx:
                X_features[:, 8 + mood_idx[mood]] = 1

            desired_mood_idx = {'Happy': 0, 'Inspired': 1, 'Excited': 2, 'Thoughtful': 3, 'Relaxed': 4, 'Thrilled': 5}
            desired_mood = str(user_preferences.get('desired_mood', 'Happy'))
            if desired_mood in desired_mood_idx:
                X_features[:, 15 + desired_mood_idx[desired_mood]] = 1

            attention_idx = {'Casual (can multitask)': 0, 'Moderate (some focus needed)': 1, 'Full attention required': 2}
            attention = str(user_preferences.get('attention_level', 'Moderate (some focus needed)'))
            if attention in attention_idx:
                X_features[:, 21 + attention_idx[attention]] = 1

            complexity_idx = {'Prefer simple plots': 0, 'Like some complexity': 1, 'Love complex mind-bending stories': 2}
            complexity = str(user_preferences.get('complexity_preference', 'Like some complexity'))
            if complexity in complexity_idx:
                X_features[:, 24 + complexity_idx[complexity]] = 1

            pacing_idx = {'Slow and steady': 0, 'Moderate pace': 1, 'Fast-paced': 2}
            pacing = str(user_preferences.get('pacing_preference', 'Moderate pace'))
            if pacing in pacing_idx:
                X_features[:, 27 + pacing_idx[pacing]] = 1

            length_idx = {'Prefer shorter movies': 0, "Don't mind longer movies": 1, 'Love epic length movies': 2}
            length = str(user_preferences.get('movie_length_preference', "Don't mind longer movies"))
            if length in length_idx:
                X_features[:, 30 + length_idx[length]] = 1

            language_idx = {'English': 0, 'Foreign language': 1, 'No preference': 2}
            language = str(user_preferences.get('language_preference', 'English'))
            if language in language_idx:
                X_features[:, 33 + language_idx[language]] = 1

            time_period_idx = {'Classic (pre-1990)': 0, 'Modern (1990-present)': 1, 'No preference': 2}
            time_period = str(user_preferences.get('time_period_preference', 'No preference'))
            if time_period in time_period_idx:
                X_features[:, 36 + time_period_idx[time_period]] = 1

            rating_pref_idx = {'Yes, highly rated only': 0, 'No preference': 1, 'Any rating': 2}
            rating_pref = str(user_preferences.get('rating_preference', 'No preference'))
            if rating_pref in rating_pref_idx:
                X_features[:, 39 + rating_pref_idx[rating_pref]] = 1

            platform_idx = {'Netflix': 0, 'Amazon Prime': 1, 'Hulu': 2, 'Disney+': 3, 'Other': 4, 'No preference': 5}
            platform = str(user_preferences.get('streaming_platform', 'No preference'))
            if platform in platform_idx:
                X_features[:, 42 + platform_idx[platform]] = 1

            y = (ratings_df['rating'] >= 4).astype(int)

            self.logger.info(
                f"Preprocessed data shapes: X_user={X_user.shape}, X_movie={X_movie.shape}, X_features={X_features.shape}, y={y.shape}"
            )
            return X_user, X_movie, X_features, y
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def train(self, X, y, epochs=10, batch_size=128):
        """
        Train the collaborative filtering model.
        
        Args:
            X (tuple): Tuple of (X_user, X_movie, X_features).
            y (np.array): Label array.
            epochs (int): Number of epochs.
            batch_size (int): Batch size.
        
        Returns:
            Training history.
        """
        try:
            if self.model is None:
                n_users = len(self.user_encoder.classes_)
                n_movies = len(self.movie_encoder.classes_)
                self.logger.info(f"Building model with {n_users} users and {n_movies} movies")
                self.build_ncf_model(n_users, n_movies)
            
            X_user, X_movie, X_features = X
            history = self.model.fit(
                [X_user, X_movie, X_features], y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=1
            )
            self.training_history = history
            self.logger.info("Model training completed successfully")
            return history
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise

    def _create_data_generator(self, X, y, batch_size):
        """
        Create a generator that yields training data batches.
        """
        def generator():
            while True:
                indices = np.random.permutation(len(y))
                for i in range(0, len(y), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    batch_X = [x[batch_indices] for x in X]
                    batch_y = y[batch_indices]
                    yield batch_X, batch_y
        return generator()

    def _validate_learning(self, history):
        """
        Validate training performance by examining final metrics.
        """
        try:
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]

            loss_diff = abs(final_train_loss - final_val_loss)
            acc_diff = abs(final_train_acc - final_val_acc)

            self.logger.info("\nLearning Validation Results:")
            self.logger.info(f"Final Training Loss: {final_train_loss:.4f}")
            self.logger.info(f"Final Validation Loss: {final_val_loss:.4f}")
            self.logger.info(f"Final Training Accuracy: {final_train_acc:.4f}")
            self.logger.info(f"Final Validation Accuracy: {final_val_acc:.4f}")

            if final_train_loss < 0.5 and final_val_loss < 0.5:
                self.logger.info("✓ Model is learning effectively (loss < 0.5)")
            else:
                self.logger.warning("⚠ Model might need more training or tuning (loss > 0.5)")
            if loss_diff > 0.2 or acc_diff > 0.2:
                self.logger.warning("⚠ Potential overfitting detected (large gap between metrics)")
            else:
                self.logger.info("✓ No significant overfitting detected")
            if final_val_acc > 0.6:
                self.logger.info("✓ Good validation accuracy achieved (> 0.6)")
            elif final_val_acc > 0.5:
                self.logger.info("✓ Decent validation accuracy achieved (> 0.5)")
            else:
                self.logger.warning("⚠ Low validation accuracy, consider adjustments")
        except Exception as e:
            self.logger.error(f"Error validating learning: {str(e)}")
            raise

    def visualize_training(self):
        """
        Generate plots for training and validation metrics over epochs.
        Saves the figure as 'training_metrics.png'.
        """
        if self.training_history is None:
            self.logger.error("No training history available")
            return

        history = self.training_history.history
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Training Metrics', fontsize=16)
        
        axes[0, 0].plot(history['loss'], label='Training Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(history['precision'], label='Training Precision')
        axes[1, 0].plot(history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(history['recall'], label='Training Recall')
        axes[1, 1].plot(history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()
        
        self.logger.info("Training visualization saved as 'training_metrics.png'")
        self.logger.info("\nFinal Training Metrics:")
        self.logger.info(f"Training Loss: {history['loss'][-1]:.4f}")
        self.logger.info(f"Training Accuracy: {history['accuracy'][-1]:.4f}")
        self.logger.info(f"Training Precision: {history['precision'][-1]:.4f}")
        self.logger.info(f"Training Recall: {history['recall'][-1]:.4f}")
        self.logger.info("\nFinal Validation Metrics:")
        self.logger.info(f"Validation Loss: {history['val_loss'][-1]:.4f}")
        self.logger.info(f"Validation Accuracy: {history['val_accuracy'][-1]:.4f}")
        self.logger.info(f"Validation Precision: {history['val_precision'][-1]:.4f}")
        self.logger.info(f"Validation Recall: {history['val_recall'][-1]:.4f}")

    def evaluate_model(self):
        """
        Evaluate the model by returning final validation metrics.
        
        Returns:
            dict: Metrics for loss, accuracy, precision, and recall.
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        try:
            metrics = {
                'val_loss': self.training_history.history['val_loss'][-1],
                'val_accuracy': self.training_history.history['val_accuracy'][-1],
                'val_precision': self.training_history.history['val_precision'][-1],
                'val_recall': self.training_history.history['val_recall'][-1]
            }
            self.logger.info("\nFinal Evaluation Metrics:")
            for metric, value in metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")
            return metrics
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise

    def predict(self, user_id, movie_id, user_preferences):
        """
        Predict a user-movie rating based on user preferences.
        
        Args:
            user_id (int): The user ID.
            movie_id (int): The movie ID.
            user_preferences (np.array): Processed user preferences vector.
        
        Returns:
            float: Prediction score.
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        try:
            X_user = self.user_encoder.transform([user_id])
            X_movie = self.movie_encoder.transform([movie_id])
            X_features = user_preferences.reshape(1, -1)
            return self.model.predict([X_user, X_movie, X_features])[0][0]
        except ValueError as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            return 0.0

    def find_similar_users(self, user_profile, user_ratings, n_similar=5):
        """
        Identify similar users based on their rating vectors and preference data.
        
        Args:
            user_profile (dict): The target user's profile.
            user_ratings (dict): The target user's ratings.
            n_similar (int): Number of similar users to return.
        
        Returns:
            list: List of tuples (other_user, similarity score).
        """
        try:
            # Load sample users data
            sample_users_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'similar_users.json')
            if os.path.exists(sample_users_path):
                with open(sample_users_path, 'r') as f:
                    sample_users = json.load(f)
                self.logger.info(f"\nLoaded {len(sample_users)} sample users from {sample_users_path}")
            else:
                self.logger.warning(f"Sample users file not found at {sample_users_path}")
                sample_users = {}

            user_rating_vector = np.zeros(len(self.movies))
            if 'watched_movies' in user_ratings and 'ratings' in user_ratings:
                for movie_id, rating in zip(user_ratings['watched_movies'], user_ratings['ratings']):
                    if movie_id in self.movie_to_idx:
                        user_rating_vector[self.movie_to_idx[movie_id]] = rating

            similarities = []
            for other_user, other_data in sample_users.items():
                if other_user != str(user_profile.get('user_id')):
                    other_vector = np.zeros(len(self.movies))
                    other_ratings = other_data.get('ratings', {})
                    if other_ratings:
                        for movie_id, rating in other_ratings.items():
                            movie_id = int(movie_id)
                            if movie_id in self.movie_to_idx:
                                other_vector[self.movie_to_idx[movie_id]] = rating
                    
                    rating_sim = cosine_similarity([user_rating_vector], [other_vector])[0][0]
                    pref_sim = self.calculate_preference_similarity(
                        user_profile.get('preferences', {}), 
                        other_data.get('preferences', {})
                    )
                    # Adjust weights to give more importance to preferences
                    total_sim = 0.4 * rating_sim + 0.6 * pref_sim
                    
                    # Only include users with similarity above threshold
                    if total_sim > 0.1:  # Lowered threshold for more matches
                        similarities.append((other_user, total_sim, other_data))

            similar_users = sorted(similarities, key=lambda x: x[1], reverse=True)[:n_similar]
            
            # Print similar users information
            if similar_users:
                self.logger.info(f"\nFound {len(similar_users)} similar users:")
                for i, (user_id, similarity, user_data) in enumerate(similar_users, 1):
                    self.logger.info(f"\nSimilar User #{i} (Similarity: {similarity:.3f}):")
                    prefs = user_data.get('preferences', {})
                    self.logger.info(f"Liked Genres: {', '.join(prefs.get('liked_genres', []))}")
                    self.logger.info(f"Disliked Genres: {', '.join(prefs.get('disliked_genres', []))}")
                    
                    # Print some of their highly rated movies
                    ratings = user_data.get('ratings', {})
                    high_rated = [(movie_id, rating) for movie_id, rating in ratings.items() if float(rating) >= 4.0]
                    if high_rated:
                        self.logger.info("Highly Rated Movies:")
                        for movie_id, rating in sorted(high_rated, key=lambda x: x[1], reverse=True)[:3]:
                            movie = self.movies[self.movies['movieId'] == int(movie_id)].iloc[0]
                            self.logger.info(f"- {movie['title']} (Rating: {rating})")
            else:
                self.logger.info("\nNo similar users found. Using content-based recommendations instead.")
            
            return [(user_id, sim) for user_id, sim, _ in similar_users]
            
        except Exception as e:
            self.logger.error(f"Error finding similar users: {str(e)}")
            return []

    def calculate_preference_similarity(self, prefs1, prefs2):
        """
        Compute similarity between two sets of user preferences.
        
        Args:
            prefs1 (dict): Preferences of user 1.
            prefs2 (dict): Preferences of user 2.
        
        Returns:
            float: Combined similarity score.
        """
        try:
            genres1 = set(prefs1.get('genres_liked', []))
            genres2 = set(prefs2.get('genres_liked', []))
            genre_sim = len(genres1.intersection(genres2)) / max(len(genres1.union(genres2)), 1)
            same_prefs = sum(1 for k in prefs1 if k != 'genres_liked' and k in prefs2 and prefs1[k] == prefs2[k])
            other_sim = same_prefs / len(prefs1)
            return 0.6 * genre_sim + 0.4 * other_sim
        except Exception as e:
            self.logger.error(f"Error calculating preference similarity: {str(e)}")
            return 0.0

    def get_collaborative_recommendations(self, user_id, user_profile, n_recommendations=10):
        """
        Get movie recommendations based on similar users.
        
        Args:
            user_id (int): The target user ID.
            user_profile (dict): The target user's profile.
            n_recommendations (int): Number of recommendations.
        
        Returns:
            list: List of (movie_id, score) tuples.
        """
        try:
            similar_users = self.find_similar_users(user_profile, self.user_ratings.get(user_id, {}))
            recommended_movies = {}
            for similar_user, similarity in similar_users:
                curr_ratings = self.user_ratings.get(similar_user, {})
                for movie_id, rating in curr_ratings.items():
                    if rating >= 4.0:
                        if movie_id not in recommended_movies:
                            recommended_movies[movie_id] = {'score': 0, 'count': 0}
                        recommended_movies[movie_id]['score'] += rating * similarity
                        recommended_movies[movie_id]['count'] += 1
            for movie_id in recommended_movies:
                recommended_movies[movie_id]['final_score'] = (
                    recommended_movies[movie_id]['score'] / recommended_movies[movie_id]['count']
                )
            watched_movies = set(user_profile.get('watched_movies', []))
            recommendations = [
                (movie_id, data['final_score'])
                for movie_id, data in recommended_movies.items()
                if movie_id not in watched_movies
            ]
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:n_recommendations]
        except Exception as e:
            self.logger.error(f"Error getting collaborative recommendations: {str(e)}")
            return []

    def get_neural_recommendations(self, user_preferences, n_recommendations=10):
        """
        Generate movie recommendations using the neural collaborative filtering model.
        
        Args:
            user_preferences (dict): Dictionary containing user preferences.
            n_recommendations (int): Number of recommendations to return.
        
        Returns:
            list: List of (movie_id, score) tuples.
        """
        try:
            self.logger.info("Generating neural recommendations...")
            user_vector = self._process_preferences(user_preferences)
            predictions = []
            for movie_id in self.movies['movieId'].values:
                genre_profile = self._get_movie_genre_profile(movie_id)
                feature_vector = np.concatenate([genre_profile, user_vector]).reshape(1, -1)
                score = self.model.predict(feature_vector)[0][0]
                predictions.append((movie_id, score))
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:n_recommendations]
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            raise

    def _get_movie_genre_profile(self, movie_id):
        """
        Get the genre profile for a given movie as an 8-dimensional vector.
        
        Args:
            movie_id (int): The movie ID.
        
        Returns:
            np.array: An 8-dimensional vector indicating genre membership.
        """
        try:
            genre_profile = np.zeros(8)
            movie = self.movies[self.movies['movieId'] == movie_id].iloc[0]
            genres = movie['genres']
            for genre in genres:
                if genre in self.genre_list:
                    idx = self.genre_list.index(genre)
                    genre_profile[idx] = 1
            return genre_profile
        except Exception as e:
            self.logger.error(f"Error getting movie genre profile: {str(e)}")
            raise

    def get_recommendations(self, user_profile, top_n=10):
        """
        Recommend movies for a user based on their processed profile.
        
        Args:
            user_profile (dict or DataFrame): User profile information.
            top_n (int): Number of recommendations.
        
        Returns:
            list: List of (movie_id, score) tuples.
        """
        try:
            # Convert user profile to the correct format
            if isinstance(user_profile, pd.DataFrame):
                user_profile = user_profile.to_dict('records')[0]
            
            # Create feature vector
            if isinstance(user_profile, dict):
                genre_profile = np.zeros(len(self.genre_list))
                preferences_vector = self._process_preferences(user_profile)
                user_profile = np.concatenate([genre_profile, preferences_vector])
            
            # Ensure correct data type
            user_profile = np.array(user_profile, dtype=np.float32)
            
            # Get valid movie IDs and create indices
            valid_movie_ids = list(self.movie_to_idx.keys())
            movie_indices = np.array([self.movie_to_idx[mid] for mid in valid_movie_ids], dtype=np.int32)
            
            # Safety check for indices
            max_index = len(self.movie_to_idx)
            if np.any(movie_indices >= max_index):
                self.logger.warning(f"Found invalid movie indices, clipping to valid range")
                movie_indices = np.clip(movie_indices, 0, max_index - 1)
            
            # Create input arrays
            user_ids = np.zeros(len(valid_movie_ids), dtype=np.int32)
            features = np.tile(user_profile, (len(valid_movie_ids), 1)).astype(np.float32)
            
            # Get predictions
            try:
                predictions = self.model.predict([user_ids, movie_indices, features])
            except Exception as e:
                self.logger.error(f"Error during model prediction: {str(e)}")
                # Fallback to content-based recommendations
                return self._get_content_based_recommendations(user_profile, top_n)
            
            # Apply temperature scaling for better diversity
            temperature = 1.5
            scaled_predictions = predictions / temperature
            
            # Create and sort recommendations
            movie_scores = list(zip(valid_movie_ids, scaled_predictions.flatten()))
            movie_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Log recommendation stats
            self.logger.info(f"Generated {len(movie_scores)} recommendations")
            self.logger.info(f"Top recommendation score: {movie_scores[0][1]:.3f}")
            
            return movie_scores[:top_n]
            
        except Exception as e:
            self.logger.error(f"Error in get_recommendations: {str(e)}")
            # Fallback to simpler recommendation method
            return self._get_content_based_recommendations(user_profile, top_n)
            
    def _get_content_based_recommendations(self, user_profile, top_n=10):
        """
        Fallback content-based recommendation method when collaborative filtering fails.
        """
        try:
            # Extract user genre preferences
            if isinstance(user_profile, np.ndarray):
                genre_preferences = user_profile[:len(self.genre_list)]
            else:
                genre_preferences = np.zeros(len(self.genre_list))
                if isinstance(user_profile, dict):
                    liked_genres = user_profile.get('preferences', {}).get('liked_genres', [])
                    for genre in liked_genres:
                        if genre in self.genre_list:
                            genre_preferences[self.genre_list.index(genre)] = 1
            
            # Calculate similarity scores
            movie_scores = []
            for movie_id in self.movies['movieId'].values:
                movie_genres = self._get_movie_genre_profile(movie_id)
                similarity = np.dot(genre_preferences, movie_genres)
                movie_scores.append((movie_id, float(similarity)))
            
            # Sort and return top recommendations
            movie_scores.sort(key=lambda x: x[1], reverse=True)
            return movie_scores[:top_n]
            
        except Exception as e:
            self.logger.error(f"Error in content-based recommendations: {str(e)}")
            return []

    def prepare_training_data(self, user_preferences, similar_users_data):
        """
        Prepare training data for the neural collaborative filtering model.
        
        This function collects user and movie IDs as well as combined feature vectors
        (60 preference features + 8 genre profile features) and binary labels.
        
        Args:
            user_preferences (dict): Current user preferences.
            similar_users_data (dict): Data of similar users.
        
        Returns:
            tuple: ([user_ids_encoded, movie_ids_encoded, features], labels)
        """
        self.logger.info("Preparing training data...")
        user_ids = []
        movie_ids = []
        features = []
        labels = []
        
        current_user_id = max(similar_users_data.keys()) + 1 if similar_users_data else 0
        all_user_ids = set([current_user_id])
        all_movie_ids = set()
        
        for user_id, user_data in similar_users_data.items():
            all_user_ids.add(user_id)
            user_ratings = user_data.get('ratings', {})
            all_movie_ids.update(user_ratings.keys())
        
        self.user_encoder.fit(list(all_user_ids))
        self.movie_encoder.fit(list(all_movie_ids))
        
        for user_id, user_data in similar_users_data.items():
            user_ratings = user_data.get('ratings', {})
            user_prefs = user_data.get('preferences', {})
            user_pref_vector = self._process_preferences(user_prefs)
            for movie_id, rating in user_ratings.items():
                if movie_id in self.movies.index:
                    genre_profile = self._get_movie_genre_profile(movie_id)
                    combined_features = np.concatenate([user_pref_vector, genre_profile])
                    user_ids.append(user_id)
                    movie_ids.append(movie_id)
                    features.append(combined_features)
                    labels.append(1 if rating >= 4 else 0)
        
        user_ids = np.array(user_ids)
        movie_ids = np.array(movie_ids)
        features = np.array(features)
        labels = np.array(labels)
        
        user_ids_encoded = self.user_encoder.transform(user_ids)
        movie_ids_encoded = self.movie_encoder.transform(movie_ids)
        
        self.logger.info(f"Prepared {len(labels)} training samples")
        self.logger.info(f"Number of unique users: {len(all_user_ids)}")
        self.logger.info(f"Number of unique movies: {len(all_movie_ids)}")
        
        return [user_ids_encoded, movie_ids_encoded, features], labels

