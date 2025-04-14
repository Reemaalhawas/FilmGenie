// src/components/MovieCard.jsx - Complete 

import React from 'react';

const MovieCard = ({ 
  title, 
  description, 
  poster, 
  rating, 
  runtime, 
  genres, 
  year, 
  cleanTitle, 
  reason 
}) => {
  // Extract year from title if not provided separately, and use cleanTitle if available
  const displayTitle = cleanTitle || title.replace(/\s*\(\d{4}\)\s*$/, '');
  const displayYear = year || (title.match(/\((\d{4})\)/) ? title.match(/\((\d{4})\)/)[1] : 'Unknown');
  
  return (
    <div className="w-[280px] sm:w-[320px] md:w-[360px] max-w-full rounded-2xl overflow-hidden shadow-lg bg-[#1c1c2e] text-white border border-[#2f2f4f]">
      {poster ? (
        <div className="w-full h-[420px] bg-black">
          <img
            src={poster}
            alt={displayTitle}
            className="w-full h-full object-contain rounded-t-2xl bg-black"
          />
        </div>
      ) : (
        <div className="w-full h-[420px] flex items-center justify-center bg-gray-700 text-sm rounded-t-2xl">
          <div className="text-center p-4">
            <p className="text-lg font-semibold mb-2">{displayTitle}</p>
            <p className="text-xs text-gray-300">No Image Available</p>
          </div>
        </div>
      )}
      <div className="p-3 space-y-2">
        <div className="flex justify-between items-center">
          <h2 className="text-lg font-semibold text-[#DFB240] truncate pr-2">{displayTitle}</h2>
          <span className="text-xs text-gray-400 whitespace-nowrap">({displayYear})</span>
        </div>
        <div className="text-sm text-gray-300 max-h-20 overflow-y-auto pr-1">
          {description}
        </div>
        <div className="flex items-center justify-between text-xs pt-1">
          <span className="text-yellow-400">⭐ {rating !== 'N/A' && rating !== 'undefined' ? rating : 'N/A'}</span>
          <span>⏱ {runtime ? `${runtime} min` : 'N/A'}</span>
        </div>
        {genres?.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-1">
            {Array.isArray(genres) ? (
              genres.map((g, i) => (
                <span
                  key={i}
                  className="bg-[#2a2a50] text-[#DFB240] px-2 py-1 text-[10px] rounded-full"
                >
                  {g}
                </span>
              ))
            ) : (
              <span className="bg-[#2a2a50] text-[#DFB240] px-2 py-1 text-[10px] rounded-full">
                {genres}
              </span>
            )}
          </div>
        )}
        {reason && (
          <div className="text-xs text-gray-400 mt-2 italic border-t border-gray-700 pt-2">
            {reason}
          </div>
        )}
      </div>
    </div>
  );
};

export default MovieCard;