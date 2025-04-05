// Home.jsx
import React from "react";

function Home({ user }) {
  return (
    <div className="home">
      <h2>Welcome to FilmGenie!</h2>
      {user ? (
        <p>Hello, {user.name}! Ready to find your next favorite movie?</p>
      ) : (
        <p>Please log in to get personalized movie recommendations.</p>
      )}
      <p>FilmGenie is your personal movie recommender. Sign in to get started.</p>
    </div>
  );
}

export default Home;


