import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const Login = () => {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleLogin = (e) => {
    e.preventDefault();

    if (username === "admin" && password === "filmgenie") {
      setError("");
      navigate("/");
    } else {
      setError("Invalid username or password. Try 'admin' / 'filmgenie'.");
    }
  };

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-gradient-to-b from-[#0E0E22] to-[#1a1a3d] text-white overflow-hidden">
      <form
        onSubmit={handleLogin}
        className="bg-gray-900 p-8 rounded-2xl shadow-lg w-96 space-y-6"
      >
        <h2 className="text-3xl font-bold text-center">Login to FilmGenie</h2>

        {error && (
          <div className="bg-red-600 text-white p-2 rounded text-sm text-center">
            {error}
          </div>
        )}

        <div>
          <label className="block text-sm font-medium mb-1">Username</label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="w-full px-4 py-2 rounded-xl bg-gray-800 border border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full px-4 py-2 rounded-xl bg-gray-800 border border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500"
            required
          />
        </div>

        <button
          type="submit"
          className="w-full py-2 bg-purple-600 hover:bg-purple-700 rounded-xl font-semibold"
        >
          Login
        </button>
      </form>
    </div>
  );
};

export default Login;
