// MovieQuiz.jsx
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';

const quizQuestions = [
  {
    id: 'mood',
    question: 'What mood are you in right now?',
    type: 'single',
    options: ['Happy', 'Sad', 'Energetic', 'Thoughtful', 'Excited', 'Calm', 'Stressed'],
  },
  {
    id: 'emotional_impact',
    question: 'How do you want the movie to make you feel?',
    type: 'single',
    options: ['Happy', 'Inspired', 'Excited', 'Thoughtful', 'Relaxed', 'Thrilled'],
  },
  {
    id: 'genres_liked',
    question: 'Which genres do you enjoy the most? (Select up to 3)',
    type: 'multiple',
    options: ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Romance', 'Thriller', 'Horror', 'Documentary'],
    limit: 3,
  },
  {
    id: 'genres_disliked',
    question: 'Are there any genres you dislike or want to avoid? (Select up to 3)',
    type: 'multiple',
    options: ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Romance', 'Thriller', 'Horror', 'Documentary'],
    limit: 3,
  },
  {
    id: 'focus_level',
    question: 'How much attention do you want to give this movie?',
    type: 'single',
    options: ['Casual (can multitask)', 'Moderate (some focus needed)', 'Full attention required'],
  },
  {
    id: 'complexity_preference',
    question: 'How do you feel about mind-bending plots?',
    type: 'single',
    options: ['Prefer simple plots', 'Like some complexity', 'Love complex mind-bending stories'],
  },
  {
    id: 'pacing_preference',
    question: 'What pacing do you prefer?',
    type: 'single',
    options: ['Slow and steady', 'Moderate pace', 'Fast-paced'],
  },
  {
    id: 'movie_length_preference',
    question: 'How do you feel about long movies (2.5+ hours)?',
    type: 'single',
    options: ['Prefer shorter movies', "Don't mind longer movies", 'Love epic length movies'],
  },
  {
    id: 'language_preference',
    question: 'Do you prefer movies in a specific language?',
    type: 'single',
    options: ['English', 'Foreign language', 'No preference'],
  },
  {
    id: 'actor_preference',
    question: 'Do you have a favorite actor or director?',
    type: 'single',
    options: ['Yes, specific actor/director', 'No preference'],
  },
  {
    id: 'director_preference',
    question: 'Do you have a favorite director?',
    type: 'single',
    options: ['Yes, specific director', 'No preference'],
  },
  {
    id: 'time_period_preference',
    question: 'Do you prefer movies from a specific time period?',
    type: 'single',
    options: ['Classic (pre-1990)', 'Modern (1990-present)', 'No preference'],
  },
  {
    id: 'rating_preference',
    question: 'Would you like recommendations based on highly-rated movies?',
    type: 'single',
    options: ['Yes, highly rated only', 'No preference'],
  },
  {
    id: 'platform_preference',
    question: 'Which streaming platforms do you use? (Select multiple)',
    type: 'multiple',
    options: ['Netflix', 'Amazon Prime', 'Hulu', 'Disney+', 'Other', 'No preference'],
    limit: 3,
  },
];

const fadeVariants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
};

const MovieQuiz = () => {
  const [step, setStep] = useState(0);
  const [responses, setResponses] = useState({});
  const navigate = useNavigate();
  const current = quizQuestions[step];

  const handleChange = (value) => {
    if (current.type === 'multiple') {
      const selected = responses[current.id] || [];
      if (selected.includes(value)) {
        setResponses({ ...responses, [current.id]: selected.filter((v) => v !== value) });
      } else if (selected.length < (current.limit || Infinity)) {
        setResponses({ ...responses, [current.id]: [...selected, value] });
      }
    } else {
      setResponses({ ...responses, [current.id]: value });
    }
  };


const handleNext = () => {
  if (step < quizQuestions.length - 1) {
    setStep(step + 1);
  } else {
    // Prepare the proper format for the backend
    const formattedResponses = {
      mood: responses.mood || '',
      desired_mood: responses.emotional_impact || '',
      genres_liked: responses.genres_liked || [],
      genres_disliked: responses.genres_disliked || [],
      attention_level: responses.focus_level || '',
      plot_complexity: responses.complexity_preference || '',
      pacing: responses.pacing_preference || '',
      movie_length: responses.movie_length_preference || '',
      language: responses.language_preference || '',
      time_period: responses.time_period_preference || '',
      rating: responses.rating_preference || '',
      streaming_platform: responses.platform_preference || ''
    };
    
    // Store the formatted responses for later use
    localStorage.setItem('filmGenieResponses', JSON.stringify(formattedResponses));
    
    // Navigate to the recommendation page
    navigate('/recommend');
  }
};

  const handleBack = () => {
    if (step > 0) setStep(step - 1);
  };

  const isSelected = (value) =>
    current.type === 'multiple'
      ? (responses[current.id] || []).includes(value)
      : responses[current.id] === value;

  return (
    <div className="fixed inset-0 bg-gradient-to-b from-[#0E0E22] to-[#1a1a3d] text-white flex flex-col items-center justify-start px-4 pt-4">
      <header className="w-full max-w-6xl flex justify-between items-center px-4 py-2 sm:px-6 sm:py-4 mb-4">
      </header>

      <div className="w-full max-w-2xl p-6 sm:p-8 rounded-xl bg-white bg-opacity-5 backdrop-blur-lg shadow-2xl">
        <div className="w-full h-2 bg-[#585D9C] rounded mb-4 overflow-hidden">
          <div
            className="h-full bg-[#DFB240] transition-all duration-300"
            style={{ width: `${((step + 1) / quizQuestions.length) * 100}%` }}
          />
        </div>
        <p className="text-sm text-[#CCCCCC] text-center mb-3">Step {step + 1} of {quizQuestions.length}</p>

        <h2 className="text-2xl sm:text-3xl font-bold text-[#DFB240] text-center mb-6">{current.question}</h2>

        <AnimatePresence mode="wait">
          <motion.div
            key={current.id}
            variants={fadeVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className={`grid gap-3 ${current.type === 'multiple' ? 'grid-cols-2 sm:grid-cols-3 text-sm' : 'grid-cols-1'}`}
          >
            {current.options.map((option) => (
              <button
                key={option}
                onClick={() => handleChange(option)}
                className={`py-3 px-5 rounded-lg border transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-[#DFB240] ${
                  isSelected(option)
                    ? 'bg-[#DFB240]/10 border-[#DFB240] text-[#DFB240]'
                    : 'border-[#585D9C] text-[#848C98] hover:bg-[#585D9C]/10'
                }`}
                aria-pressed={isSelected(option)}
              >
                {option}
              </button>
            ))}
          </motion.div>
        </AnimatePresence>

        <div className="mt-8 flex justify-between items-center">
          <button
            onClick={handleBack}
            disabled={step === 0}
            className="px-4 py-2 rounded bg-[#585D9C] text-white disabled:opacity-40 focus:outline-none"
          >
            Back
          </button>
          <button
            onClick={handleNext}
            className="px-4 py-2 rounded bg-[#DFB240] text-black hover:bg-[#e5a900] transition-colors focus:outline-none"
          >
            {step === quizQuestions.length - 1 ? 'Finish' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default MovieQuiz;