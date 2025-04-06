import { Link } from "react-router-dom"
import "./App.css"

function Home() {
  return (
    <div className="home-container">
      <div className="hero-section">
        <h1>Welcome to Genie</h1>
        <p className="hero-subtitle">Your destination for the best movie ideas</p>

        <div className="hero-description">
          <p>
            Discover movies tailored to your taste. Get personalized recommendations based on your preferences and
            viewing history.
          </p>
        </div>

        <div className="cta-buttons">
          <Link to="/login" className="cta-button primary">
            Log In
          </Link>
          <Link to="/signup" className="cta-button secondary">
            Sign Up
          </Link>
        </div>
      </div>
    </div>
  )
}

export default Home

