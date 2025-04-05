# config.py
import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "super-secret-key")  # for session cookies

    # OAuth credentials (set in environment for security)
    GOOGLE_OAUTH_CLIENT_ID = os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "<YOUR_GOOGLE_CLIENT_ID>")
    GOOGLE_OAUTH_CLIENT_SECRET = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "<YOUR_GOOGLE_CLIENT_SECRET>")
    OAUTHLIB_RELAX_TOKEN_SCOPE = True  # allow different scopes from Google
    OAUTHLIB_INSECURE_TRANSPORT = True  # ONLY for local development (HTTP)

    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "sqlite:///filmgenie.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

