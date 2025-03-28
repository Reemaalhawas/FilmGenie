import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import LabelEncoder
import os

class MovieRecommender:
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.model = None
        
    def build_ncf_model(self, num_users, num_movies, embedding_dim=16):
        # User embedding
        user_input = layers.Input(shape=(1,), name='user_input')
        user_embedding = layers.Embedding(num_users, embedding_dim, name='user_embedding')(user_input)
        user_embedding = layers.Flatten()(user_embedding)
        
        # Movie embedding
        movie_input = layers.Input(shape=(1,), name='movie_input')
        movie_embedding = layers.Embedding(num_movies, embedding_dim, name='movie_embedding')(movie_input)
        movie_embedding = layers.Flatten()(movie_embedding)
        
        # Additional features (genres, mood, etc.)
        feature_input = layers.Input(shape=(28,), name='feature_input')  # Updated to match our feature vector size
        feature_dense = layers.Dense(64, activation='relu')(feature_input)  # Increased size to handle more features
        
        # Concatenate embeddings and features
        concat = layers.Concatenate()([user_embedding, movie_embedding, feature_dense])
        
        # Neural layers
        dense1 = layers.Dense(128, activation='relu')(concat)  # Increased size
        dense2 = layers.Dense(64, activation='relu')(dense1)   # Increased size
        output = layers.Dense(1, activation='sigmoid')(dense2)
        
        # Create model
        self.model = Model(inputs=[user_input, movie_input, feature_input], outputs=output)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    def preprocess_data(self, ratings_df, movies_df, user_preferences_df):
        # Load and preprocess movies.dat
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
        
        # Load and preprocess ratings from CSV
        ratings_df = pd.read_csv('../data/ratings_updated.csv')
        
        # Rename columns to match expected format
        column_mapping = {
            'UserID': 'userId',
            'MovieID': 'movieId',
            'Rating': 'rating'
        }
        ratings_df = ratings_df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_columns = ['userId', 'movieId', 'rating']
        if not all(col in ratings_df.columns for col in required_columns):
            raise ValueError(f"Ratings file must contain columns: {required_columns}")
        
        # Encode users and movies
        self.user_encoder.fit(ratings_df['userId'].unique())
        self.movie_encoder.fit(movies_df['movieId'].unique())
        
        # Create training data
        X_user = self.user_encoder.transform(ratings_df['userId'])
        X_movie = self.movie_encoder.transform(ratings_df['movieId'])
        
        # Use Like_Dislike column directly if available, otherwise convert rating to binary
        if 'Like_Dislike' in ratings_df.columns:
            y = ratings_df['Like_Dislike'].values
        else:
            y = (ratings_df['rating'] >= 4).astype(int)
        
        # Create feature matrix by repeating user preferences for each rating
        # Convert preferences to numerical format first
        preferences = user_preferences_df.iloc[0]  # Get first row since we only have one user
        feature_vector = []
        
        # Mood encoding
        mood_encoding = {'Happy': 0, 'Sad': 1, 'Energetic': 2, 'Thoughtful': 3, 'Excited': 4, 'Calm': 5, 'Stressed': 6}
        feature_vector.append(mood_encoding[preferences['mood']])
        
        # Desired mood encoding
        desired_mood_encoding = {'Happy': 0, 'Inspired': 1, 'Excited': 2, 'Thoughtful': 3, 'Relaxed': 4, 'Thrilled': 5}
        feature_vector.append(desired_mood_encoding[preferences['desired_mood']])
        
        # Genres (multi-hot)
        genre_encoding = {'Action': 0, 'Comedy': 1, 'Drama': 2, 'Sci-Fi': 3, 'Romance': 4, 'Thriller': 5, 'Horror': 6, 'Documentary': 7}
        liked_genres = [0] * 8
        for genre in preferences['genres_liked']:
            liked_genres[genre_encoding[genre]] = 1
        feature_vector.extend(liked_genres)
        
        disliked_genres = [0] * 8
        for genre in preferences['genres_disliked']:
            disliked_genres[genre_encoding[genre]] = 1
        feature_vector.extend(disliked_genres)
        
        # Attention level
        attention_encoding = {'Casual (can multitask)': 0, 'Moderate (some focus needed)': 1, 'Full attention required': 2}
        feature_vector.append(attention_encoding[preferences['attention_level']])
        
        # Complexity preference
        complexity_encoding = {'Prefer simple plots': 0, 'Like some complexity': 1, 'Love complex mind-bending stories': 2}
        feature_vector.append(complexity_encoding[preferences['complexity_preference']])
        
        # Pacing preference
        pacing_encoding = {'Slow and steady': 0, 'Moderate pace': 1, 'Fast-paced': 2}
        feature_vector.append(pacing_encoding[preferences['pacing_preference']])
        
        # Movie length preference
        length_encoding = {'Prefer shorter movies': 0, 'Don\'t mind longer movies': 1, 'Love epic length movies': 2}
        feature_vector.append(length_encoding[preferences['movie_length_preference']])
        
        # Language preference
        language_encoding = {'English': 0, 'Foreign language': 1, 'No preference': 2}
        feature_vector.append(language_encoding[preferences['language_preference']])
        
        # Actor preference
        actor_encoding = {'Yes, specific actor/director': 0, 'No preference': 1}
        feature_vector.append(actor_encoding[preferences['actor_preference']])
        
        # Director preference
        director_encoding = {'Yes, specific director': 0, 'No preference': 1}
        feature_vector.append(director_encoding[preferences['director_preference']])
        
        # Time period preference
        time_encoding = {'Classic (pre-1990)': 0, 'Modern (1990-present)': 1, 'No preference': 2}
        feature_vector.append(time_encoding[preferences['time_period_preference']])
        
        # Rating preference
        rating_encoding = {'Yes, highly rated only': 0, 'No preference': 1}
        feature_vector.append(rating_encoding[preferences['rating_preference']])
        
        # Streaming preference
        streaming_encoding = {'Netflix': 0, 'Amazon Prime': 1, 'Hulu': 2, 'Disney+': 3, 'Other': 4, 'No preference': 5}
        feature_vector.append(streaming_encoding[preferences['streaming_preference']])
        
        # Convert to numpy array and repeat for each rating
        X_features = np.tile(np.array(feature_vector, dtype=np.float32), (len(ratings_df), 1))
        
        return X_user, X_movie, X_features, y
    
    def train(self, X_user, X_movie, X_features, y, epochs=10, batch_size=64):
        if self.model is None:
            self.build_ncf_model(
                num_users=len(self.user_encoder.classes_),
                num_movies=len(self.movie_encoder.classes_)
            )
        
        self.model.fit(
            [X_user, X_movie, X_features],
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
    
    def predict(self, user_id, movie_id, user_preferences):
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        X_user = self.user_encoder.transform([user_id])
        X_movie = self.movie_encoder.transform([movie_id])
        X_features = user_preferences.reshape(1, -1)
        
        return self.model.predict([X_user, X_movie, X_features])[0][0]
    
    def get_recommendations(self, user_id, user_preferences, top_n=10):
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Get all movies
        all_movies = self.movie_encoder.classes_
        
        # Predict ratings for all movies
        predictions = []
        for movie_id in all_movies:
            pred = self.predict(user_id, movie_id, user_preferences)
            predictions.append((movie_id, pred))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n] 