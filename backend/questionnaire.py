def get_user_preferences():
    print("\nPlease answer the following questions to personalize your movie recommendations.\n")

    preferences = {}

    preferences['mood'] = input("What's your current mood? (e.g., Happy, Sad, Excited, Relaxed, Thoughtful): ").strip()
    preferences['desired_mood'] = input("What kind of feeling do you want from the movie? (e.g., Happy, Sad, Excited, Relaxed, Thoughtful): ").strip()

    print("\nList your favorite genres separated by commas (e.g., Action, Comedy, Drama): ")
    preferences['genres_liked'] = [genre.strip() for genre in input().split(',')]

    print("\nList any genres you dislike (e.g., Horror, Romance). Leave blank if none: ")
    disliked = input().strip()
    preferences['genres_disliked'] = [genre.strip() for genre in disliked.split(',')] if disliked else []

    preferences['attention_level'] = input("What's your attention level? (Casual, Moderate, Full): ").strip()
    preferences['plot_complexity'] = input("Do you prefer simple, moderate, or complex plots? ").strip()
    preferences['pacing'] = input("Do you prefer slow, medium, or fast-paced movies? ").strip()
    preferences['movie_length'] = input("Do you prefer short, medium, or long movies? ").strip()
    preferences['language'] = input("Preferred language? (English, Foreign, Both): ").strip()
    preferences['time_period'] = input("Preferred time period? (Classic, Modern, Contemporary, Any, Specific): ").strip()
    preferences['rating'] = input("Any rating preference? (G, PG, PG-13, R, Any): ").strip()
    preferences['streaming_platform'] = input("Preferred streaming platform? (Netflix, Amazon, Hulu, Disney+, Other): ").strip()

    print("\nThanks! Generating your movie preferences...\n")
    return preferences