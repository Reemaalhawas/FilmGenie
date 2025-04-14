import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';

import LandingPage from './components/LandingPage';
import MovieQuiz from './components/MovieQuiz';
import Results from './components/Results';
import MovieSwiper from './components/MovieSwiper';
import Login from './pages/Login';
import Navbar from './components/Navbar';

const AppContent = () => {
  const location = useLocation();

  return (
    <div className="flex flex-col min-h-screen">
      <Navbar />
      <div className="flex flex-1">
        <div className="flex-1">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/quiz" element={<MovieQuiz />} />
            <Route path="/results" element={<Results />} />
            <Route path="/recommend" element={<MovieSwiper />} />
            <Route path="/login" element={<Login />} />
          </Routes>
        </div>
      </div>
    </div>
  );
};

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;
