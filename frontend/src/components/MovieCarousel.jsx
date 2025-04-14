import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import MovieCard from './MovieCard';
import { useNavigate } from 'react-router-dom';

const movieData = [
  {
    id: 1,
    title: 'Inception (2025)',
    description: 'A thief who steals corporate secrets through dream-sharing technology.',
    rating: 8.8,
    runtime: 148,
    genres: ['Sci-Fi', 'Thriller', 'Action'],
    poster: null,
  },
  {
    id: 2,
    title: 'The Matrix (2025)',
    description: 'A hacker discovers reality is a simulation and joins the rebellion.',
    rating: 7.2,
    runtime: 136,
    genres: ['Sci-Fi', 'Action'],
    poster: null,
  },
  {
    id: 3,
    title: 'Interstellar (2025)',
    description: 'Explorers travel through a wormhole in search of a new home for humanity.',
    rating: 9.1,
    runtime: 169,
    genres: ['Sci-Fi', 'Adventure', 'Drama'],
    poster: null,
  },
];

const swipeConfidenceThreshold = 100;

const MovieSwiper = ({ onComplete }) => {
  const [movies, setMovies] = useState(movieData);
  const [index, setIndex] = useState(0);
  const [swipes, setSwipes] = useState([]);
  const [showTip, setShowTip] = useState(true);
  const [swipeDirection, setSwipeDirection] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (index >= 3) setShowTip(false);
  }, [index]);

  const finish = (finalSwipes) => {
    localStorage.setItem('filmGenieRated', JSON.stringify(finalSwipes));
    onComplete?.(finalSwipes);
    navigate('/results');
  };

  const handleSwipe = (direction) => {
    const currentMovie = movies[index];
    if (direction === 'up') {
      advance(swipes);
      return;
    }

    const liked = direction === 'right';
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

      <button onClick={skipMovie} className="mt-6 px-5 py-2 rounded bg-[#585D9C] text-white hover:bg-[#6c6fbc] transition">Skip</button>

      <div className="absolute bottom-6 text-sm text-[#848C98]">
        {index + 1} / {movies.length}
      </div>
    </div>
  );
};

export default MovieSwiper;
