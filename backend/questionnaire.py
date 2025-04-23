def get_user_preferences():
    print("\nPlease answer the following questions to personalize your movie recommendations.\n")

    preferences = {}

    def get_single_input(prompt, options=None):
        while True:
            answer = input(prompt).strip()
            if not options or answer in options:
                return answer
            print(f"Invalid input. Please choose from: {', '.join(options)}")

    genre_options = [
        "Action", "Comedy", "Drama", "Horror", "Romance", "Thriller",
        "Sci-Fi", "Fantasy", "Documentary", "Mystery", "Adventure"
    ]

    preferences['mood'] = get_single_input(
        "What's your current mood? (e.g., Happy, Sad, Excited, Relaxed, Thoughtful): "
    )
    
    preferences['desired_mood'] = get_single_input(
        "What kind of feeling do you want from the movie? (e.g., Happy, Sad, Excited, Relaxed, Thoughtful): "
    )

    preferences['genres_liked'] = get_single_input(
        f"Pick your favorite genre ({', '.join(genre_options)}): ",
        genre_options
    )

    genre_options_with_none = genre_options + ["None"]
    disliked = get_single_input(
        f"Pick one genre you dislike or type 'None' if none ({', '.join(genre_options_with_none)}): ",
        genre_options_with_none
    )
    preferences['genres_disliked'] = None if disliked == "None" else disliked

    preferences['attention_level'] = get_single_input(
        "What's your attention level? (Casual, Moderate, Full): ",
        ["Casual", "Moderate", "Full"]
    )

    preferences['plot_complexity'] = get_single_input(
        "Do you prefer simple, moderate, or complex plots? ",
        ["simple", "moderate", "complex"]
    )

    preferences['pacing'] = get_single_input(
        "Do you prefer slow, medium, or fast-paced movies? ",
        ["slow", "medium", "fast"]
    )

    preferences['movie_length'] = get_single_input(
        "Do you prefer short, medium, or long movies? ",
        ["short", "medium", "long"]
    )

    preferences['language'] = get_single_input(
        "Preferred language? (English, Foreign, Both): ",
        ["English", "Foreign", "Both"]
    )

    preferences['time_period'] = get_single_input(
        "Preferred time period? (Classic, Modern, Contemporary, Any): ",
        ["Classic", "Modern", "Contemporary", "Any"]
    )

    print("\nThanks! Generating your movie preferences...\n")
    return preferences
