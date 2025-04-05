# app.py
from flask import Flask, send_from_directory
from config import Config
from database import db
from auth import login_manager, google_bp, auth_bp
from recommendations import reco_bp

from flask_cors import CORS  # enable if frontend runs on a different domain (for dev)

def create_app():
    app = Flask(__name__, static_folder="../frontend/build", static_url_path="/")
    app.config.from_object(Config)

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)

    # Set up user loader for Flask-Login
    from models import User
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Configure Google OAuth blueprint with credentials from config
    google_bp.client_id = app.config["GOOGLE_OAUTH_CLIENT_ID"]
    google_bp.client_secret = app.config["GOOGLE_OAUTH_CLIENT_SECRET"]

    # Register blueprints
    app.register_blueprint(google_bp, url_prefix="/login")    # Flask-Dance Google OAuth routes
    app.register_blueprint(auth_bp, url_prefix="/auth")       # Our auth-related routes (logout, after_login)
    app.register_blueprint(reco_bp)                           # Recommendations and API routes (no prefix, so /api/* as defined)

    # Enable CORS for API routes if needed (so React dev server can call them)
    CORS(app, resources={r"/api/*": {"origins": "*"}})  # adjust origins in production for security

    # Serve React frontend (assuming build files are in frontend/build)
    @app.route('/')
    def serve_index():
        """Serve the React app's index.html."""
        return send_from_directory(app.static_folder, 'index.html')
    # Also handle serving static assets (JS, CSS, images) from React build
    @app.errorhandler(404)
    def not_found(e):
        # If a route is not found in Flask, serve the React index (for client-side routing)
        return send_from_directory(app.static_folder, 'index.html')

    return app

# Running the app (for local development)
if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        db.create_all()  # ensure database tables are created
    app.run(debug=True)
