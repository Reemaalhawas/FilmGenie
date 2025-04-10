import json
import random

def generate_sample_data():
    """Generate sample user data and save to JSON file"""
    # Option lists for individual user preferences
    moods = ["Happy", "Sad", "Excited", "Relaxed", "Thoughtful"]
    desired_feelings = ["Happy", "Sad", "Excited", "Relaxed", "Thoughtful"]
    liked_genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Documentary", "Animation", "Fantasy"]
    attention_levels = ["Casual", "Moderate", "Full"]
    plot_complexities = ["Simple", "Moderate", "Complex"]
    pacings = ["Slow", "Medium", "Fast"]
    movie_lengths = ["Short", "Medium", "Long"]
    languages = ["English", "Foreign", "Both"]
    time_periods = ["Classic", "Modern", "Contemporary", "Any", "Specific"]
    rating_options = ["G", "PG", "PG-13", "R", "Any"]
    streaming_platforms = ["Netflix", "Amazon", "Hulu", "Disney+", "Other"]

    # Generate sample_users.json (list of 100 individual user records)
    sample_users = []
    for user_id in range(1, 101):
        # Random number of movies watched
        num_movies = random.randint(3, 5)
        # Generate a random list of movie IDs (for example purposes, using numbers 100 to 600)
        watched_movies = random.sample(range(100, 601), num_movies)
        # Corresponding ratings (values between 1 and 5)
        ratings = [random.randint(1, 5) for _ in range(num_movies)]
        
        user_profile = {
            "user_id": user_id,
            "watched_movies": watched_movies,
            "ratings": ratings,
            "preferences": {
                "mood": random.choice(moods),
                "desired_feeling": random.choice(desired_feelings),
                "liked_genres": random.sample(liked_genres, random.randint(1, 3)),
                "disliked_genres": random.sample(liked_genres, random.randint(0, 2)),
                "attention_level": random.choice(attention_levels),
                "plot_complexity": random.choice(plot_complexities),
                "pacing": random.choice(pacings),
                "movie_length": random.choice(movie_lengths),
                "language": random.choice(languages),
                "time_period": random.choice(time_periods),
                "rating": random.choice(rating_options),
                "streaming_platform": random.choice(streaming_platforms)
            }
        }
        sample_users.append(user_profile)

    # Write the JSON data to file
    with open("backend/sample_users.json", "w") as f_users:
        json.dump(sample_users, f_users, indent=4)

    print("Generated sample_users.json with 100 users.")

if __name__ == "__main__":
    generate_sample_data() 