import os
import requests
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
ADMIN_PASSWORD       = os.getenv("ADMIN_PASSWORD", "admin123")

AUTH_BASE    = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL    = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


def get_google_auth_url(redirect_uri: str) -> str:
    if not GOOGLE_CLIENT_ID:
        raise ValueError("GOOGLE_CLIENT_ID not set in .env")
    params = {
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  redirect_uri,
        "response_type": "code",
        "scope":         "openid email profile",
        "access_type":   "offline",
        "prompt":        "consent",
    }
    return f"{AUTH_BASE}?{urlencode(params)}"


def get_user_info(code: str, redirect_uri: str):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise ValueError("Google credentials not set in .env")
    token_data = {
        "code":          code,
        "client_id":     GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri":  redirect_uri,
        "grant_type":    "authorization_code",
    }
    try:
        token_resp = requests.post(TOKEN_URL, data=token_data, timeout=10)
        token_resp.raise_for_status()
        token_json = token_resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Token exchange failed: {e}")
        return None
    access_token = token_json.get("access_token")
    if not access_token:
        print(f"[ERROR] No access token: {token_json}")
        return None
    try:
        user_resp = requests.get(
            USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        user_resp.raise_for_status()
        return user_resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] User info fetch failed: {e}")
        return None


def check_admin_password(password: str) -> bool:
    return password == ADMIN_PASSWORD
