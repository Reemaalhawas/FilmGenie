// App.jsx
import { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate } from "react-router-dom";
import Login from "./Login";
import Home from "./Home";
import MovieRecommendations from "./MovieRecommendations";
import Onboarding from "./Onboarding";
import Navbar from "./Navbar";

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(null);  // null = not checked yet
  const [userProfile, setUserProfile] = useState(null);

  // Check session on initial load
  useEffect(() => {
    fetch("/api/profile")
      .then(res => res.json())
      .then(data => {
        if (data.logged_in) {
          setIsAuthenticated(true);
          setUserProfile(data.user);  // data.user might contain name/email
        } else {
          setIsAuthenticated(false);
        }
      })
      .catch(err => {
        console.error("Error checking profile:", err);
        setIsAuthenticated(false);
      });
  }, []);

  const handleLogout = () => {
    // Call logout API and update state
    fetch("/auth/logout")
      .then(() => {
        setIsAuthenticated(false);
        setUserProfile(null);
      })
      .catch(err => console.error("Logout failed", err));
  };

  if (isAuthenticated === null) {
    // Show a loading indicator while we determine auth status
    return <div>Loading...</div>;
  }

  return (
    <Router>
      <Navbar isAuthenticated={isAuthenticated} onLogout={handleLogout} />
      <Routes>
        <Route path="/" element={<Home user={userProfile} />} />
        <Route 
          path="/login" 
          element={isAuthenticated ? <Navigate to="/recommendations" /> : <Login />} 
        />
        <Route 
          path="/recommendations" 
          element={
            isAuthenticated ? <MovieRecommendations /> : <Navigate to="/login" />
          } 
        />
        <Route 
          path="/onboarding" 
          element={
            isAuthenticated ? <Onboarding onComplete={() => { /* handled in Onboarding */ }} /> 
                            : <Navigate to="/login" />
          } 
        />
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </Router>
  );
}

export default App;

