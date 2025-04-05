// Login.jsx
import React from "react";
import "./Auth.css";  // assume some styling for auth buttons/containers

function Login() {
  const handleGoogleLogin = () => {
    // Redirect the browser to the Flask OAuth login route
    window.location.href = "/login/google";
  };

  return (
    <div className="auth-container">
      <div className="auth-form-container">
        <h2>Log In to FilmGenie</h2>
        <button className="google-button" onClick={handleGoogleLogin}>
          <img src="https://developers.google.com/identity/images/g-logo.png" alt="Google logo" />
          Sign in with Google
        </button>
      </div>
    </div>
  );
}

export default Login;


