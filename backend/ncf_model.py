import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotNormal
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieRecommender:
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        self.model = None
        self.training_history = None
        
    def build_ncf_model(self, num_users, num_movies, embedding_dim=32):
        try:
            logger.info(f"Building model with {num_users} users and {num_movies} movies")
            
            # User embedding
            user_input = layers.Input(shape=(1,), name='user_input')
            user_embedding = layers.Embedding(
                num_users, 
                embedding_dim, 
                name='user_embedding',
                embeddings_initializer=GlorotNormal()
            )(user_input)
            user_embedding = layers.Flatten()(user_embedding)
            user_embedding = layers.BatchNormalization()(user_embedding)
            
            # Movie embedding
            movie_input = layers.Input(shape=(1,), name='movie_input')
            movie_embedding = layers.Embedding(
                num_movies, 
                embedding_dim, 
                name='movie_embedding',
                embeddings_initializer=GlorotNormal()
            )(movie_input)
            movie_embedding = layers.Flatten()(movie_embedding)
            movie_embedding = layers.BatchNormalization()(movie_embedding)
            
            # Additional features
            feature_input = layers.Input(shape=(28,), name='feature_input')
            feature_dense = layers.Dense(128, activation='relu', kernel_initializer=GlorotNormal())(feature_input)
            feature_dense = layers.BatchNormalization()(feature_dense)
            feature_dense = layers.Dropout(0.3)(feature_dense)
            
            # Concatenate embeddings and features
            concat = layers.Concatenate()([user_embedding, movie_embedding, feature_dense])
            
            # Neural layers with proper initialization and regularization
            dense1 = layers.Dense(256, activation='relu', kernel_initializer=GlorotNormal())(concat)
            dense1 = layers.BatchNormalization()(dense1)
            dense1 = layers.Dropout(0.3)(dense1)
            
            dense2 = layers.Dense(128, activation='relu', kernel_initializer=GlorotNormal())(dense1)
            dense2 = layers.BatchNormalization()(dense2)
            dense2 = layers.Dropout(0.2)(dense2)
            
            dense3 = layers.Dense(64, activation='relu', kernel_initializer=GlorotNormal())(dense2)
            dense3 = layers.BatchNormalization()(dense3)
            dense3 = layers.Dropout(0.1)(dense3)
            
            output = layers.Dense(1, activation='sigmoid')(dense3)
            
            # Create model
            self.model = Model(inputs=[user_input, movie_input, feature_input], outputs=output)
            
            # Compile with proper learning rate and metrics
            optimizer = Adam(learning_rate=0.001)
            self.model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )
            
            logger.info("Model built and compiled successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            return False
        
    def preprocess_data(self, ratings_df, movies_df, user_preferences_df):
        try:
            logger.info("Starting data preprocessing")
            
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
                
            # Check for NaN values in ratings
            if ratings_df['rating'].isna().any():
                logger.warning("Found NaN values in ratings. Removing them...")
                ratings_df = ratings_df.dropna(subset=['rating'])
                
            # Check for invalid ratings
            if not all(0 <= r <= 5 for r in ratings_df['rating']):
                logger.warning("Found invalid ratings. Removing them...")
                ratings_df = ratings_df[ratings_df['rating'].between(0, 5)]
            
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
                
            # Check for NaN values in target
            if np.isnan(y).any():
                logger.warning("Found NaN values in target variable. Removing them...")
                valid_indices = ~np.isnan(y)
                X_user = X_user[valid_indices]
                X_movie = X_movie[valid_indices]
                y = y[valid_indices]
            
            # Process user preferences
            preferences = user_preferences_df.iloc[0]
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
            
            # Scale features
            X_features = self.feature_scaler.fit_transform(X_features)
            
            logger.info(f"Data preprocessing completed. Shapes: X_user={X_user.shape}, X_movie={X_movie.shape}, X_features={X_features.shape}, y={y.shape}")
            return X_user, X_movie, X_features, y
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def train(self, X_user, X_movie, X_features, y, epochs=50, batch_size=128):
        try:
            if self.model is None:
                success = self.build_ncf_model(
                    num_users=len(self.user_encoder.classes_),
                    num_movies=len(self.movie_encoder.classes_)
                )
                if not success:
                    raise ValueError("Failed to build model")
            
            # Add validation data
            validation_split = 0.2
            val_size = int(len(X_user) * validation_split)
            
            # Shuffle the data
            indices = np.random.permutation(len(X_user))
            X_user = X_user[indices]
            X_movie = X_movie[indices]
            X_features = X_features[indices]
            y = y[indices]
            
            # Split into train and validation
            X_user_train, X_user_val = X_user[:-val_size], X_user[-val_size:]
            X_movie_train, X_movie_val = X_movie[:-val_size], X_movie[-val_size:]
            X_features_train, X_features_val = X_features[:-val_size], X_features[-val_size:]
            y_train, y_val = y[:-val_size], y[-val_size:]
            
            # Define callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=1e-6,
                    verbose=1
                ),
                TensorBoard(
                    log_dir='./logs',
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True
                )
            ]
            
            logger.info("Starting model training...")
            logger.info(f"Training data shapes: X_user_train={X_user_train.shape}, X_movie_train={X_movie_train.shape}, X_features_train={X_features_train.shape}, y_train={y_train.shape}")
            logger.info(f"Validation data shapes: X_user_val={X_user_val.shape}, X_movie_val={X_movie_val.shape}, X_features_val={X_features_val.shape}, y_val={y_val.shape}")
            
            # Train the model with validation data and callbacks
            self.training_history = self.model.fit(
                [X_user_train, X_movie_train, X_features_train],
                y_train,
                validation_data=([X_user_val, X_movie_val, X_features_val], y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("Training completed successfully")
            return self.training_history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def evaluate_model(self):
        if self.training_history is None:
            raise ValueError("Model has not been trained yet")
            
        # Plot training history
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history.history['loss'], label='Training Loss')
        plt.plot(self.training_history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.training_history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print final metrics
        final_metrics = {
            'Training Loss': self.training_history.history['loss'][-1],
            'Validation Loss': self.training_history.history['val_loss'][-1],
            'Training Accuracy': self.training_history.history['accuracy'][-1],
            'Validation Accuracy': self.training_history.history['val_accuracy'][-1]
        }
        
        logger.info("Final Model Metrics:")
        for metric, value in final_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
        return final_metrics
    
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