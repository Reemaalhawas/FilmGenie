import React from "react";
import { Link, useLocation } from "react-router-dom";

const Navbar = () => {
  const location = useLocation();

  const isActive = (path) =>
    location.pathname === path ? "text-yellow-300 font-semibold" : "";

  return (
    <nav className="w-full bg-[#2C235A] py-4 px-6 flex justify-between items-center text-white font-medium shadow-md z-50">
      <div className="text-xl font-bold tracking-wide">
        <Link to="/">
        <div className="flex items-center space-x-3">
          <img src="/FilmGenieLogo.jpg" alt="Film Genie Logo" className="w-10 h-10 rounded-full" />
          <span className="text-2xl font-bold animate-pulse">Film Genie</span>
        </div>
        </Link>
      </div>
      <div className="space-x-6 text-sm sm:text-base">
        <Link to="/" className={`hover:text-yellow-300 transition ${isActive("/")}`}>Home</Link>
        <Link to="/quiz" className={`hover:text-yellow-300 transition ${isActive("/quiz")}`}>Quiz</Link>
        <Link to="/recommend" className={`hover:text-yellow-300 transition ${isActive("/recommend")}`}>Recommend</Link>
        <Link to="/results" className={`hover:text-yellow-300 transition ${isActive("/results")}`}>Results</Link>
        <Link to="/login" className={`hover:text-yellow-300 transition ${isActive("/login")}`}>Login</Link>
      </div>
    </nav>
  );
};

export default Navbar;
