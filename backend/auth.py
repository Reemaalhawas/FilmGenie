# auth.py
from flask import Blueprint, url_for, redirect, session
from flask_dance.contrib.google import make_google_blueprint, google
from flask_login import LoginManager, login_user, logout_user, current_user
from models import User, db

# Initialize Flask-Login manager (to be used in app.py)
login_manager = LoginManager()
login_manager.login_view = 'google.login'  # not used directly since we use Google OAuth

# Setup Google OAuth blueprint
google_bp = make_google_blueprint(
    client_id=None, client_secret=None,
    scope=["profile", "email"],  # request access to basic profile info and email
    redirect_to="auth.after_login"
)
# Note: We'll set client_id and secret from app config in app.py after blueprint creation.

auth_bp = Blueprint('auth', __name__)

@auth_bp.route("/logout")
def logout():
    """Log out the current user."""
    logout_user()
    # Optionally, revoke Google token if needed
    session.clear()
    return {"message": "Logged out"}, 200

@auth_bp.route("/after_login")
def after_login():
    """
    This view is called after Google OAuth dance is complete.
    Flask-Dance will redirect here if redirect_to is set to 'auth.after_login'.
    """
    if not google.authorized:
        return redirect(url_for("google.login"))  # If not authorized, try again or go to login

    # Fetch user info from Google
    resp = google.get("/oauth2/v2/userinfo")
    # If using Google OAuth v1 endpoints, use google.get("/plus/v1/people/me") or similar
    if not resp.ok:
        return "Failed to fetch user info from Google.", 500
    user_info = resp.json()
    # user_info will contain keys like "id", "email", "name", "picture"
    google_id = user_info["id"]
    email = user_info.get("email")
    name = user_info.get("name")
    profile_pic = user_info.get("picture")

    # Sign in or create the user
    user = User.query.filter_by(google_id=google_id).first()
    if user is None:
        # New user: create user record
        user = User(google_id=google_id, email=email, name=name, profile_pic=profile_pic)
        db.session.add(user)
        db.session.commit()
    else:
        # Existing user: update info in case it changed
        user.email = email
        user.name = name
        user.profile_pic = profile_pic
        db.session.commit()

    # Log the user in (Flask-Login)
    login_user(user)
    # At this point, user is logged in with session cookie

    # Redirect to frontend page (e.g., recommendations or onboarding check)
    # We return a simple HTML redirect or JSON, since this route is hit by the browser after OAuth.
    # Here, redirect to the root (“/”) – the React app will handle routing post-login.
    return redirect("/")
