import React from 'react';
import { NavLink } from 'react-router-dom';

const Sidebar = () => {
  const linkClass = 'block px-4 py-2 rounded hover:bg-blue-800 transition text-white';
  const activeClass = 'bg-blue-700 font-bold';

  return (
    <div className="w-60 bg-blue-900 text-white min-h-screen fixed top-0 left-0 shadow-lg flex flex-col justify-between">
      <div>
        <div className="p-6 text-2xl font-bold border-b border-blue-700">🎥 Film Genie</div>
        <nav className="flex flex-col p-4 space-y-2">
          <NavLink to="/" className={({ isActive }) => `${linkClass} ${isActive ? activeClass : ''}`}>
            🏠 Home
          </NavLink>
          <NavLink to="/quiz" className={({ isActive }) => `${linkClass} ${isActive ? activeClass : ''}`}>
            🧠 Movie Quiz!
          </NavLink>
          <NavLink to="/recommend" className={({ isActive }) => `${linkClass} ${isActive ? activeClass : ''}`}>
            🎬 Movie picker!
          </NavLink>
        </nav>
      </div>
    </div>
  );
};

export default Sidebar;
