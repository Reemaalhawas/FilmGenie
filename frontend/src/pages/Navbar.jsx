// Navbar.jsx
import React from "react";
import { Link } from "react-router-dom";

function Navbar({ isAuthenticated, onLogout }) {
  return (
    <nav className="navbar">
      <h1 className="logo">🎥 FilmGenie</h1>
      <ul>
        {isAuthenticated ? (
          <>
            <li><Link to="/recommendations">Recommendations</Link></li>
            <li><Link to="/#" onClick={(e) => { e.preventDefault(); onLogout(); }}>Logout</Link></li>
          </>
        ) : (
          <li><Link to="/login">Login</Link></li>
        )}
      </ul>
    </nav>
  );
}

export default Navbar;
