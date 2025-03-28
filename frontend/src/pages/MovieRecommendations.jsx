"use client"

import { useState } from "react"
import "./style.css"

// Mock data for movie recommendations
const mockMovies = [
  {
    id: 1,
    title: "Inception",
    year: 2010,
    genre: "Sci-Fi",
    director: "Christopher Nolan",
    poster: "https://via.placeholder.com/150x225?text=Inception",
    rating: 8.8,
    description:
      "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.",
  },
  {
    id: 2,
    title: "The Shawshank Redemption",
    year: 1994,
    genre: "Drama",
    director: "Frank Darabont",
    poster: "https://via.placeholder.com/150x225?text=Shawshank",
    rating: 9.3,
    description:
      "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
  },
  {
    id: 3,
    title: "The Dark Knight",
    year: 2008,
    genre: "Action",
    director: "Christopher Nolan",
    poster: "https://via.placeholder.com/150x225?text=Dark+Knight",
    rating: 9.0,
    description:
      "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
  },
  {
    id: 4,
    title: "Pulp Fiction",
    year: 1994,
    genre: "Crime",
    director: "Quentin Tarantino",
    poster: "https://via.placeholder.com/150x225?text=Pulp+Fiction",
    rating: 8.9,
    description:
      "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
  },
  {
    id: 5,
    title: "The Godfather",
    year: 1972,
    genre: "Crime",
    director: "Francis Ford Coppola",
    poster: "https://via.placeholder.com/150x225?text=Godfather",
    rating: 9.2,
    description:
      "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
  },
  {
    id: 6,
    title: "Fight Club",
    year: 1999,
    genre: "Drama",
    director: "David Fincher",
    poster: "https://via.placeholder.com/150x225?text=Fight+Club",
    rating: 8.8,
    description:
      "An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much, much more.",
  },
]

function MovieRecommendations() {
  const [movies] = useState(mockMovies)
  const [selectedGenre, setSelectedGenre] = useState("All")
  const [searchTerm, setSearchTerm] = useState("")

  // Genres for filter
  const genres = ["All", "Action", "Drama", "Sci-Fi", "Crime"]

  // Filter movies based on genre and search term
  const filteredMovies = movies.filter((movie) => {
    const matchesGenre = selectedGenre === "All" || movie.genre === selectedGenre
    const matchesSearch =
      movie.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      movie.director.toLowerCase().includes(searchTerm.toLowerCase())
    return matchesGenre && matchesSearch
  })

  return (
    <div className="recommendations-container">
      <h1>Movie Recommendations</h1>

      <div className="filter-container">
        <div className="search-box">
          <input
            type="text"
            placeholder="Search movies or directors..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        <div className="genre-filter">
          <label>Filter by Genre:</label>
          <select value={selectedGenre} onChange={(e) => setSelectedGenre(e.target.value)}>
            {genres.map((genre) => (
              <option key={genre} value={genre}>
                {genre}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="movies-grid">
        {movies.map((movie) => (
          <div key={movie.id} className="movie-card">
            <h3>{movie.title}</h3>
            <p>Genre: {movie.genre}</p>
            <p>Rating: {movie.rating}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default MovieRecommendations

