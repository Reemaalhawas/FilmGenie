"use client"

import { Link } from "react-router-dom"
import "./App.css"

function Navbar({ isAuthenticated, onLogout }) {
  return (
    <nav className="navbar">
      <div className="navbar-logo">
        <Link to="/">
          <h1>Genie</h1>
        </Link>
      </div>
      <div className="navbar-links">
        {isAuthenticated ? (
          <>
            <Link to="/recommendations" className="nav-link">
              Recommendations
            </Link>
            <button onClick={onLogout} className="nav-link logout-btn">
              Logout
            </button>
          </>
        ) : (
          <>
            <Link to="/login" className="nav-link">
              Login
            </Link>
            <Link to="/signup" className="nav-link signup-btn">
              Sign Up
            </Link>
          </>
        )}
      </div>
    </nav>
  )
}

export default Navbar

