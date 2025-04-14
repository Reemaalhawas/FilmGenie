// src/components/Results.jsx - Complete replacement

import React, { useEffect, useState } from 'react';
import MovieCard from './MovieCard';
import { getRecommendations, enrichMovieWithTMDB } from '../services/api';

const Results = () => {
  const [recommendations, setRecommendations] = useState([]);
  const [userProfile, setUserProfile] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const [source, setSource] = useState("ai-model"); // Now always AI-model

  const fetchRecommendations = async () => {
    try {
      setIsLoading(true);
      
      // Get quiz responses from localStorage
      const quiz = JSON.parse(localStorage.getItem('filmGenieResponses') || '{}');
      
      // Set user profile data for UI
      setUserProfile({
        mood: quiz.mood,
        vibe: quiz.desired_mood || quiz.vibe,
        genres: quiz.genres_liked || [],
      });
      
      // Get recommendations from backend
      const backendRecommendations = await getRecommendations(quiz);
      
      if (backendRecommendations && backendRecommendations.length > 0) {
        console.log("Processing backend recommendations:", backendRecommendations);
        
        // Enrich with TMDB data for display
        const formattedRecs = await Promise.all(
          backendRecommendations.map(enrichMovieWithTMDB)
        );
        
        setRecommendations(formattedRecs);
      } else {
        // No recommendations
        console.warn("No recommendations received from backend");
        setRecommendations([]);
      }
    } catch (error) {
      console.error("Error fetching recommendations:", error);
      setRecommendations([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchRecommendations();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#0E0E22] to-[#1a1a3d] text-white px-4 py-8">
      {isLoading ? (
        <p className="text-center text-[#CCCCCC]">Finding the perfect movies for your vibe...</p>
      ) : (
        <>
          {/* Watchlist Heading + Summary */}
          <div className="text-center max-w-2xl mx-auto mb-12">
            <h2 className="text-3xl font-bold text-[#DFB240] flex items-center justify-center gap-2">
              🎥 <span>Your Watchlist</span>
            </h2>
            <p className="text-[#CCCCCC] italic mt-2">
              {`You're in a ${userProfile.mood || 'curious'} mood, vibing ${userProfile.vibe || 'Mystery'}, loving genres like ${
                userProfile.genres?.length ? userProfile.genres.join(', ') : 'Drama'
              }.`}
            </p>
            <p className="text-[#DFB240] text-sm mt-2">✨ AI-Powered Recommendations ✨</p>
          </div>

          <div className="flex flex-wrap justify-center gap-6">
            {recommendations.length > 0 ? (
              recommendations.map(movie => (
                <MovieCard
                  key={movie.id}
                  title={movie.title}
                  poster={movie.poster}
                  description={movie.description}
                  genres={movie.genres}
                  reason={movie.reason}
                  rating={movie.rating}
                  runtime={movie.runtime}
                  year={movie.year}
                  cleanTitle={movie.cleanTitle}
                />
              ))
            ) : (
              <div className="text-center max-w-lg mx-auto">
                <p className="text-[#CCCCCC] mb-4">
                  No recommendations found that match your preferences.
                </p>
                <p className="text-[#CCCCCC]">
                  Try updating your quiz answers or swiping on more movies.
                </p>
              </div>
            )}
          </div>
        </>
      )}

      {/* Genie Analysis */}
      {!isLoading && recommendations.length > 0 && (
        <div className="mt-16 max-w-3xl mx-auto bg-white/5 p-6 rounded-xl border border-[#585D9C] shadow text-center">
          <h2 className="text-2xl font-bold text-[#DFB240] mb-4">🧞 Genie Analysis</h2>
          <p className="text-[#CCCCCC] mb-4">Here's how the Genie conjured your recommendations:</p>

          <div className="grid sm:grid-cols-3 gap-6 text-left text-sm text-white">
            <div>
              <p className="font-semibold text-[#DFB240] mb-1">Your Mood</p>
              <p className="text-[#CCCCCC]">{userProfile.mood || 'Unknown'}</p>
            </div>
            <div>
              <p className="font-semibold text-[#DFB240] mb-1">Your Vibe</p>
              <p className="text-[#CCCCCC]">{userProfile.vibe || 'Mystery'}</p>
            </div>
            <div>
              <p className="font-semibold text-[#DFB240] mb-1">Your Favorite Genres</p>
              <div className="flex flex-wrap gap-2 mt-1">
                {userProfile.genres?.map((genre, i) => (
                  <span key={i} className="px-2 py-1 text-xs border border-[#DFB240] text-[#DFB240] rounded-full bg-[#DFB240]/10">
                    {genre}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-6 text-sm text-[#CCCCCC] italic">
            🧠 These picks are powered by our neural network model trained on thousands of users' preferences.
          </div>
        </div>
      )}
    </div>
  );
};

export default Results;