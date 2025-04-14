// getTMDBRecommendations.js
export async function getTMDBRecommendations(likedMovies) {
    const apiKey = process.env.REACT_APP_TMDB_API_KEY;
    if (!likedMovies || likedMovies.length === 0) return [];
  
    const movieIDs = likedMovies.map(movie => movie.id);
  
    const fetchSimilar = async (id) => {
      const res = await fetch(`https://api.themoviedb.org/3/movie/${id}/similar?api_key=${apiKey}`);
      const data = await res.json();
      return data.results || [];
    };
  
    let recommendations = [];
  
    for (const id of movieIDs) {
      try {
        const similar = await fetchSimilar(id);
        recommendations = [...recommendations, ...similar];
      } catch (err) {
        console.error(`Failed to fetch similar movies for ID ${id}`, err);
      }
    }
  
    // Remove duplicates by TMDB ID
    const unique = {};
    recommendations.forEach(movie => {
      if (!unique[movie.id]) {
        unique[movie.id] = {
          id: movie.id,
          title: movie.title,
          description: movie.overview,
          poster: movie.poster_path
            ? `https://image.tmdb.org/t/p/w500${movie.poster_path}`
            : null,
        };
      }
    });
  
    return Object.values(unique).slice(0, 5); // Limit to 20 fresh recs
  }
  