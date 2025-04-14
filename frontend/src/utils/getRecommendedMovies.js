// getRecommendedMovies.js
export function getRecommendedMovies(likedMovies) {
  if (!likedMovies || likedMovies.length === 0) return [];

  const likedTitles = likedMovies.map(m => m.title.toLowerCase());

  const allMovies = [
    { title: 'Tenet', description: 'A secret agent embarks on a dangerous mission using time inversion.' },
    { title: 'Blade Runner 2049', description: 'A new blade runner unearths secrets that could plunge society into chaos.' },
    { title: 'Arrival', description: 'A linguist helps communicate with aliens after they arrive on Earth.' },
    { title: 'The Prestige', description: 'Two rival magicians battle to create the ultimate illusion.' },
    { title: 'Looper', description: 'A hitman faces his older self in a time-travel showdown.' },
  ];

  // Recommend all that weren’t already liked
  return allMovies.filter(movie => !likedTitles.includes(movie.title.toLowerCase()));
}
