# models.py
from database import db
from flask_login import UserMixin
import json

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    google_id = db.Column(db.String(255), unique=True, nullable=False)    # Google unique user ID
    email = db.Column(db.String(255), unique=True, nullable=False)
    name = db.Column(db.String(255))
    profile_pic = db.Column(db.String(512))
    preferences_json = db.Column(db.Text, nullable=True)  # store onboarding answers as JSON (if any)

    def set_preferences(self, prefs: dict):
        """Store user preferences (from questionnaire) as JSON text."""
        self.preferences_json = json.dumps(prefs)

    def get_preferences(self):
        """Retrieve preferences as dict (or None if not set)."""
        if not self.preferences_json:
            return None
        return json.loads(self.preferences_json)
