// src/services/api.js - Complete replacement

const API_BASE_URL = 'http://localhost:5000/api';
const TMDB_API_KEY = process.env.REACT_APP_TMDB_API_KEY;

// Clear session before starting any new recommendation process
export const clearSession = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/clear-session`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    return await response.json();
  } catch (error) {
    console.error('Error clearing session:', error);
    return { status: 'error' };
  }
};

// Get swipe candidates from our backend (movies to swipe on)
export const getSwipeCandidates = async (preferences) => {
  try {
    console.log("Fetching swipe candidates with preferences:", preferences);
    
    const response = await fetch(`${API_BASE_URL}/swipe-candidates`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(preferences),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log("Received swipe candidates:", data);
    
    if (data.status === 'success' && data.candidates && data.candidates.length > 0) {
      return data.candidates;
    } else {
      console.warn("No swipe candidates received from backend");
      return [];
    }
  } catch (error) {
    console.error('Error fetching swipe candidates:', error);
    return [];
  }
};

// Record a swipe in our backend
export const recordSwipe = async (movieId, liked) => {
  try {
    const response = await fetch(`${API_BASE_URL}/swipe`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ movieId, liked }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error recording swipe:', error);
    // Silently fail
    return { status: 'error' };
  }
};

// Get recommendations from our backend
export const getRecommendations = async (preferences) => {
  try {
    console.log("Sending preferences to API:", preferences);
    
    // First clear any previous session data
    await clearSession();
    
    // Get swipe data from localStorage
    const swipes = JSON.parse(localStorage.getItem('filmGenieRated') || '[]');
    
    // Create the data to send, combining preferences and swipe data
    const requestData = {
      ...preferences,
      swipes: swipes.map(movie => ({
        movieId: movie.id,
        liked: movie.liked
      }))
    };
    
    // Request recommendations
    const response = await fetch(`${API_BASE_URL}/recommend`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log("Received recommendations:", data);
    
    if (data.status === 'success') {
      return data.recommendations || [];
    } else {
      console.warn("API returned error:", data.message);
      return [];
    }
  } catch (error) {
    console.error('Error getting recommendations:', error);
    return [];
  }
};

// Lookup movie in TMDB to get poster and additional info
export const lookupMovieInTMDB = async (title) => {
  try {
    // Search for the movie by title
    const searchUrl = `https://api.themoviedb.org/3/search/movie?api_key=${TMDB_API_KEY}&query=${encodeURIComponent(title)}`;
    const searchResponse = await fetch(searchUrl);
    
    if (!searchResponse.ok) {
      throw new Error(`TMDB search failed: ${searchResponse.status}`);
    }
    
    const searchData = await searchResponse.json();
    
    // Return first result if any
    if (searchData.results && searchData.results.length > 0) {
      const movieId = searchData.results[0].id;
      
      // Get detailed movie info
      const detailUrl = `https://api.themoviedb.org/3/movie/${movieId}?api_key=${TMDB_API_KEY}`;
      const detailResponse = await fetch(detailUrl);
      
      if (!detailResponse.ok) {
        return searchData.results[0]; // Return basic info if details fail
      }
      
      return await detailResponse.json();
    }
    
    return null;
  } catch (error) {
    console.error('Error looking up movie in TMDB:', error);
    return null;
  }
};

// Enrich our backend movie with TMDB data
export const enrichMovieWithTMDB = async (movie) => {
  try {
    const searchTitle = movie.CleanTitle || movie.Title.replace(/\s*\(\d{4}\)\s*$/, '');
    const tmdbMovie = await lookupMovieInTMDB(searchTitle);
    
    if (!tmdbMovie) {
      return {
        id: movie.MovieID,
        title: movie.Title,
        description: `A ${movie.Genres} movie.`,
        poster: null,
        rating: "N/A",
        runtime: null,
        genres: movie.Genres ? movie.Genres.split('|') : [],
        year: movie.Year,
        cleanTitle: movie.CleanTitle
      };
    }
    
    return {
      id: movie.MovieID,
      title: movie.Title,
      description: tmdbMovie.overview || `A ${movie.Genres} movie.`,
      poster: tmdbMovie.poster_path ? `https://image.tmdb.org/t/p/w500${tmdbMovie.poster_path}` : null,
      rating: tmdbMovie.vote_average?.toFixed(1) || "N/A",
      runtime: tmdbMovie.runtime || null,
      genres: movie.Genres ? movie.Genres.split('|') : [],
      year: movie.Year,
      cleanTitle: movie.CleanTitle
    };
  } catch (error) {
    console.error('Error enriching movie with TMDB data:', error);
    
    // Return basic info if TMDB lookup fails
    return {
      id: movie.MovieID,
      title: movie.Title,
      description: `A ${movie.Genres} movie.`,
      poster: null,
      rating: "N/A",
      runtime: null,
      genres: movie.Genres ? movie.Genres.split('|') : [],
      year: movie.Year,
      cleanTitle: movie.CleanTitle
    };
  }
};