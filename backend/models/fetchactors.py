import pandas as pd
import requests
import time
import re
from fuzzywuzzy import process, fuzz

# Your TMDb API key – ensure this key is valid for TMDb
tmdb_api_key = "003adae1101937665573ff312074cf78"

# Define the path using a raw string
file_path = r"C:\Users\reema\FilmGenie\data\movies.dat"

# Load the dataset
try:
    movies = pd.read_csv(file_path, sep="::", engine="python", encoding="latin-1", 
                         header=None, names=["MovieID", "Title", "Genres"])
    print("Dataset loaded successfully:")
    print(movies.head())
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

def fix_title(title):
    """
    Reformat titles that end with a trailing article.
    E.g., "City of Lost Children, The" becomes "The City of Lost Children".
    """
    match = re.match(r"^(.*),\s+(The|A|An)$", title, re.IGNORECASE)
    if match:
        return f"{match.group(2)} {match.group(1)}"
    return title

def extract_title_and_year(full_title):
    """
    Extract a clean title and release year from a full title string.
    Expected format: "Title (Year)" (with optional trailing articles).
    Returns a tuple: (clean_title, year) where year can be None.
    """
    match = re.match(r"^(.*)\s+\((\d{4})\)$", full_title)
    if match:
        raw_title = match.group(1).strip()
        year = match.group(2)
        clean_title = fix_title(raw_title)
        # Remove any special characters or additional info
        clean_title = re.sub(r"[^\w\s]", "", clean_title)
        return clean_title, year
    else:
        return fix_title(full_title), None

def get_movie_id_fuzzy(title, year, retries=3, threshold=50):
    """
    Search TMDb for a movie using its title (and year if available) and perform
    fuzzy matching on the returned results to select the best match.
    Returns the TMDb movie ID if a match with a score above 'threshold' is found.
    """
    search_url = "https://api.themoviedb.org/3/search/movie"
    search_params = {"api_key": tmdb_api_key, "query": title}
    if year:
        search_params["year"] = year
        
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(search_url, params=search_params, timeout=10)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            print(f"TMDb: Results for '{title} ({year})': {results}")  # Debugging info
            if not results and year:
                print(f"TMDb: No results for '{title} ({year})'. Trying without year...")
                search_params.pop("year", None)
                response = requests.get(search_url, params=search_params, timeout=10)
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])
                print(f"TMDb: Results for '{title}' without year: {results}")  # Debugging info
                if not results:
                    print(f"TMDb: No results for '{title}' even without year.")
                    return None
            result_titles = [(res["title"], res["id"]) for res in results]
            best_match = process.extractOne(title, [rt[0] for rt in result_titles], scorer=fuzz.token_set_ratio)
            if best_match:
                print(f"TMDb: Best match for '{title}' is '{best_match[0]}' with score {best_match[1]}")
                if best_match[1] >= threshold:
                    for t, movie_id in result_titles:
                        if t == best_match[0]:
                            return movie_id
                else:
                    print(f"TMDb: Fuzzy matching did not yield a good match for '{title}'. Best score: {best_match[1]}")
                    return None
            else:
                print(f"TMDb: No fuzzy match result for '{title}'")
                return None
        except Exception as e:
            print(f"TMDb: Attempt {attempt} - Error searching for '{title} ({year})': {e}")
            time.sleep(1)
    return None

def get_movie_actors_tmdb(full_title, retries=3):
    """
    Given a movie's full title, use fuzzy matching to get the TMDb movie ID and then
    retrieve its credits, returning a comma-separated string of actor names.
    """
    title, year = extract_title_and_year(full_title)
    movie_id = get_movie_id_fuzzy(title, year, retries)
    if not movie_id:
        print(f"TMDb: Could not find a TMDb ID for '{full_title}'")
        return None

    credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
    credits_params = {"api_key": tmdb_api_key}
    
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(credits_url, params=credits_params, timeout=10)
            response.raise_for_status()
            data = response.json()
            cast_list = data.get("cast", [])
            actor_names = [member["name"] for member in cast_list]
            print(f"TMDb: Actors for '{full_title}': {actor_names}")  # Debugging info
            return ", ".join(actor_names)
        except Exception as e:
            print(f"TMDb: Attempt {attempt} - Error fetching credits for '{full_title}': {e}")
            time.sleep(2)  # Add a delay between retries
    return None

def get_movie_keywords_tmdb(full_title, retries=3):
    """
    Given a movie's full title, use fuzzy matching to get the TMDb movie ID and then
    retrieve its keywords, returning a comma-separated string of keywords.
    """
    title, year = extract_title_and_year(full_title)
    movie_id = get_movie_id_fuzzy(title, year, retries)
    if not movie_id:
        print(f"TMDb: Could not find a TMDb ID for '{full_title}'")
        return None

    keywords_url = f"https://api.themoviedb.org/3/movie/{movie_id}/keywords"
    keywords_params = {"api_key": tmdb_api_key}
    
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(keywords_url, params=keywords_params, timeout=10)
            response.raise_for_status()
            data = response.json()
            keywords_list = data.get("keywords", [])
            keywords = [keyword["name"] for keyword in keywords_list]
            print(f"TMDb: Keywords for '{full_title}': {keywords}")  # Debugging info
            return ", ".join(keywords)
        except Exception as e:
            print(f"TMDb: Attempt {attempt} - Error fetching keywords for '{full_title}': {e}")
            time.sleep(2)  # Add a delay between retries
    return None

# Apply the functions to fetch actors and keywords
movies["Actors"] = movies["Title"].apply(get_movie_actors_tmdb)
movies["Keywords"] = movies["Title"].apply(get_movie_keywords_tmdb)

# Print the updated dataset with actor and keyword information
print("Updated dataset with actor and keyword information from TMDb:")
print(movies[["Title", "Actors", "Keywords"]].head())

# Save the updated dataset to a new file
output_file_path = r"C:\Users\reema\FilmGenie\data\movies_with_actors_and_keywords.csv"
movies.to_csv(output_file_path, index=False)
print(f"Dataset with actors and keywords saved to {output_file_path}")