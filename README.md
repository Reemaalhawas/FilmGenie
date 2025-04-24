
#  FilmGenie

A hybrid movie recommendation system developed as part of the ECS537U - Design and Build Project in Artificial Intelligence module at Queen Mary University of London.

## Overview

FilmGenie combines collaborative filtering and content-based filtering to deliver personalized movie recommendations. The system analyzes both user behavior patterns and movie content to provide accurate and diverse suggestions.

### Key Components:
- **Collaborative Filtering**: Analyzes user behavior and preferences
- **Content-Based Filtering**: Examines movie features and characteristics
- **Hybrid Model**: Combines both approaches for enhanced recommendations

##  Features

-  Interactive movie quiz for preference collection
-  Modern, responsive user interface
-  Real-time movie swiping interface
- TMDB API integration


##  Getting Started

### Prerequisites

- Node.js (v14+)
- Python 3.8+
- npm/yarn
- TMDB API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Reemaalhawas/FilmGenie.git
cd FilmGenie
```

2. Backend setup:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

3. Frontend setup:
```bash
cd frontend
npm install
```

4. Configure environment:
Create `.env` in frontend directory:
```
REACT_APP_TMDB_API_KEY=your_tmdb_api_key
REACT_APP_API_URL=http://localhost:5000
```

### Running the Application

1. Start backend:
```bash
python main.py
```

2. Start frontend:
```bash
cd frontend
npm start
```

Access at: `http://localhost:3000`

## Technical Stack

### Frontend
- React.js
- Tailwind CSS
- React Router


### Backend
- Python
- Flask
- TensorFlow/Keras
- Pandas
- NumPy
- Scikit-learn

### APIs
- TMDB API
- Custom Recommendation API


##  Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/`)
3. Commit changes (`git commit -m ''`)
4. Push to branch (`git push origin feature/`)
5. Open Pull Request



Acknowledgments

- [TMDB](https://www.themoviedb.org/) for movie database API
- Queen Mary University of London
- All contributors


