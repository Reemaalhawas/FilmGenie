// src/components/MovieSwiper.jsx - Complete replacement

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import MovieCard from './MovieCard';
import { getSwipeCandidates, recordSwipe, lookupMovieInTMDB } from '../services/api';

const swipeConfidenceThreshold = 100;

const MovieSwiper = () => {
  const [movies, setMovies] = useState([]);
  const [index, setIndex] = useState(0);
  const [swipes, setSwipes] = useState([]);
  const [showTip, setShowTip] = useState(true);
  const [isLoading, setIsLoading] = useState(true);
  const [swipeDirection, setSwipeDirection] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchMovies = async () => {
      setIsLoading(true);
      
      try {
        // Get preferences from quiz
        const preferences = JSON.parse(localStorage.getItem('filmGenieResponses') || '{}');
        
        // Fetch movies from our backend
        const candidates = await getSwipeCandidates(preferences);
        
        if (candidates && candidates.length > 0) {
          // Enrich with TMDB data for display
          const enrichedMovies = await Promise.all(candidates.map(async (movie) => {
            try {
              // Get title for searching (without year if possible)
              const searchTitle = movie.CleanTitle || 
                movie.Title.replace(/\s*\(\d{4}\)\s*$/, '');
                
              // Look up in TMDB for poster and additional info
              const tmdbMovie = await lookupMovieInTMDB(searchTitle);
              
              return {
                id: movie.MovieID,
                title: movie.Title,
                description: tmdbMovie?.overview || `A ${movie.Genres} movie.`,
                poster: tmdbMovie?.poster_path ? 
                  `https://image.tmdb.org/t/p/w500${tmdbMovie.poster_path}` : null,
                rating: tmdbMovie?.vote_average?.toFixed(1) || "N/A",
                runtime: tmdbMovie?.runtime || null,
                genres: movie.Genres ? movie.Genres.split('|') : [],
                year: movie.Year,
                cleanTitle: movie.CleanTitle
              };
            } catch (error) {
              console.error(`Error enriching movie ${movie.Title}:`, error);
              
              // Return basic info if TMDB lookup fails
              return {
                id: movie.MovieID,
                title: movie.Title,
                description: `A ${movie.Genres} movie.`,
                poster: null,
                rating: "N/A",
                runtime: null,
                genres: movie.Genres ? movie.Genres.split('|') : []
              };
            }
          }));
          
          setMovies(enrichedMovies);
        } else {
          console.warn("No movies received from backend");
          setMovies([]);
        }
      } catch (error) {
        console.error("Error fetching movies:", error);
        setMovies([]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchMovies();
  }, []);

  useEffect(() => {
    if (index >= 3) setShowTip(false);
  }, [index]);

  const finish = (finalSwipes) => {
    // Store ratings in localStorage
    localStorage.setItem('filmGenieRated', JSON.stringify(finalSwipes));
    // Navigate to results
    navigate('/results');
  };

  const handleSwipe = async (direction) => {
    if (!movies[index]) return;
    
    const currentMovie = movies[index];
    
    if (direction === 'up') {
      // Skip - don't record any preference
      advance(swipes);
      return;
    }

    const liked = direction === 'right';
    
    // Record swipe in backend
    try {
      await recordSwipe(currentMovie.id, liked);
    } catch (error) {
      console.error("Error recording swipe:", error);
      // Continue even if recording fails
    }
    
    const newSwipes = [...swipes, { ...currentMovie, liked }];
    advance(newSwipes);
  };

  const advance = (updatedSwipes) => {
    setSwipeDirection(null);
    if (index + 1 >= movies.length) {
      finish(updatedSwipes);
    } else {
      setIndex(index + 1);
      setSwipes(updatedSwipes);
    }
  };

  const handleDragEnd = (event, info) => {
    const { x, y } = info.offset;
    if (x > swipeConfidenceThreshold) {
      setSwipeDirection('right');
      handleSwipe('right');
    } else if (x < -swipeConfidenceThreshold) {
      setSwipeDirection('left');
      handleSwipe('left');
    } else if (y < -swipeConfidenceThreshold) {
      setSwipeDirection('up');
      handleSwipe('up');
    }
  };

  const skipMovie = () => handleSwipe('up');

  return (
    <div className="fixed inset-0 bg-gradient-to-b from-[#0E0E22] to-[#1a1a3d] text-white flex flex-col items-center justify-center px-4 overflow-hidden">
      <header className="absolute top-4 left-4 flex items-center space-x-3 z-10">
        <img src="/FilmGenieLogo.jpg" alt="Logo" className="w-10 h-10 rounded-full" />
        <span className="text-xl font-playfair font-bold text-[#DFB240] animate-pulse">Film Genie</span>
      </header>

      {/* Loading state */}
      {isLoading ? (
        <div className="text-center">
          <p className="text-xl text-[#DFB240] mb-2">Gathering movies for you...</p>
          <p className="text-gray-400">Based on your quiz preferences</p>
        </div>
      ) : movies.length === 0 ? (
        <div className="text-center">
          <p className="text-xl text-[#DFB240] mb-2">No movies found</p>
          <p className="text-gray-400">Try different preferences in the quiz</p>
          <button 
            onClick={() => navigate('/quiz')}
            className="mt-6 px-5 py-2 rounded bg-[#585D9C] text-white hover:bg-[#6c6fbc] transition"
          >
            Back to Quiz
          </button>
        </div>
      ) : (
        <>
          <AnimatePresence>
            {showTip && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.4 }}
                className="absolute z-20 max-w-xs w-[90%] bg-[#1c1c2e] text-[#DFB240] border border-[#DFB240] rounded-xl px-5 py-4 text-center shadow-xl"
                style={{ top: '50%', transform: 'translateY(-50%)' }}
              >
                <button onClick={() => setShowTip(false)} className="absolute top-1 right-2 text-[#DFB240] text-sm hover:text-white transition">✕</button>
                <p className="text-sm font-medium leading-relaxed">
                  👉 Swipe Left = Nope <br />
                  👈 Swipe Right = Yes! <br />
                  🔼 Swipe Up = Skip
                </p>
              </motion.div>
            )}
          </AnimatePresence>

          <div className="relative w-full max-w-md h-[90vh] flex items-center justify-center">
            <AnimatePresence>
              {movies[index] && (
                <motion.div
                  key={movies[index].id}
                  drag
                  dragConstraints={{ top: 0, bottom: 0, left: 0, right: 0 }}
                  onDragEnd={handleDragEnd}
                  initial={{ opacity: 0, y: 50 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.9, y: -30 }}
                  transition={{ duration: 0.3 }}
                  className="absolute"
                >
                  <div className="relative">
                    <MovieCard {...movies[index]} />
                    {swipeDirection === 'left' && <span className="absolute top-4 left-4 text-red-500 text-2xl font-bold">NOPE</span>}
                    {swipeDirection === 'right' && <span className="absolute top-4 right-4 text-green-500 text-2xl font-bold">YES!</span>}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <button 
            onClick={skipMovie} 
            className="mt-6 px-5 py-2 rounded bg-[#585D9C] text-white hover:bg-[#6c6fbc] transition"
          >
            Skip
          </button>

          <div className="absolute bottom-6 text-sm text-[#848C98]">
            {index + 1} / {movies.length}
          </div>
        </>
      )}
    </div>
  );
};

export default MovieSwiper;