"use client"

import { useState } from "react"
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom"
import "./App.css"
import Login from "./Login"
import SignUpForm from "./SignUpForm"
import Home from "./Home"
import MovieRecommendations from "./MovieRecommendations"
import Navbar from "./Navbar"

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)

  const handleLogin = () => {
    setIsAuthenticated(true)
  }

  const handleLogout = () => {
    setIsAuthenticated(false)
  }

  return (
    <Router>
      <div className="app-container">
        <Navbar isAuthenticated={isAuthenticated} onLogout={handleLogout} />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route
            path="/login"
            element={isAuthenticated ? <Navigate to="/recommendations" /> : <Login onLogin={handleLogin} />}
          />
          <Route
            path="/signup"
            element={isAuthenticated ? <Navigate to="/recommendations" /> : <SignUpForm onSignup={handleLogin} />}
          />
          <Route
            path="/recommendations"
            element={isAuthenticated ? <MovieRecommendations /> : <Navigate to="/login" />}
          />
        </Routes>
      </div>
    </Router>
  )
}

export default App

