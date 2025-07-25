import json
import logging
import os
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pathlib
import pytz
import streamlit as st

import google.auth
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from dateutil import parser

import sqlite3
import logging

# ---------------------------------------------------------------------------
# Secret management helper (must be defined before use)
# ---------------------------------------------------------------------------

def get_secret(key: str, default: str | None = ""):
    """Return secret value from OS env first, then Streamlit secrets.

    Designed for server deployments (e.g. Railway) where secrets are passed
    via environment variables. Falls back to `st.secrets` when running the
    Streamlit front-end locally or on Streamlit Cloud.
    """
    # 1) Environment variable (preferred for Railway / Docker)
    env_val = os.getenv("key")
    if env_val not in (None, ""):
        return env_val

    # 2) Streamlit secrets (used when running in Streamlit Cloud or locally)
    try:
        import streamlit as _st  # local import to avoid heavy dependency at runtime

        if key in _st.secrets:
            return _st.secrets[key]
        if "default" in _st.secrets and key in _st.secrets["default"]:
            return _st.secrets["default"][key]
    except ModuleNotFoundError:
        # Streamlit not installed – ignore
        pass
    except Exception:
        # Any secrets access error – ignore and use default
        pass

    return default

# ---------------------------------------------------------------------------
# Module setup
# ---------------------------------------------------------------------------

from .database import (
    user_credentials,
    cleanup_expired_credentials,
    load_user_credentials,
    store_user_credentials,
    delete_user_credentials,
)

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar"]
# Resolve project root (two levels up from this file)
_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
# Allow override via environment variable; otherwise look in root dir
CREDENTIALS_FILE = os.path.abspath(
    os.getenv("GOOGLE_CREDENTIALS_FILE", str(_ROOT_DIR / "credentials.json"))
)

# Prefer explicit environment variable, else fall back to Streamlit secrets or default URL
REDIRECT_URI = os.getenv(
    "OAUTH_REDIRECT_URI",
    get_secret("OAUTH_REDIRECT_URI", "https://ai-booking-agent-jut8.onrender.com/callback"),
)

# ---------------------------------------------------------------------------
# Ensure credentials file exists from environment secret
# ---------------------------------------------------------------------------

def _ensure_credentials_file() -> None:
    """Write credentials.json from secrets/env if it is missing."""
    if os.path.exists(CREDENTIALS_FILE):
        return
    
    logger.info("credentials.json not found, attempting to create from secrets...")
    
    creds_blob: dict | None = None
    
    # 1) Attempt to load GOOGLE_CREDENTIALS_JSON from secrets/env
    creds_json_str = get_secret("GOOGLE_CREDENTIALS_JSON", None)
    if creds_json_str:
        try:
            creds_blob = json.loads(creds_json_str) if isinstance(creds_json_str, str) else creds_json_str
            logger.info("GOOGLE_CREDENTIALS_JSON successfully loaded from secret/env var")
        except Exception as e:
            logger.error("Failed to parse GOOGLE_CREDENTIALS_JSON: %s", e)
    
    # Try to construct from individual secrets if full blob is missing or invalid
    if not creds_blob:
        logger.info("Attempting to build credentials from individual OAuth secrets...")
        try:
            # Get OAuth components
            client_id = get_secret('GOOGLE_OAUTH_CLIENT_ID')
            project_id = get_secret('GOOGLE_OAUTH_PROJECT_ID')
            auth_uri = get_secret('GOOGLE_OAUTH_AUTH_URI') or "https://accounts.google.com/o/oauth2/auth"
            token_uri = get_secret('GOOGLE_OAUTH_TOKEN_URI') or "https://oauth2.googleapis.com/token"
            certs_uri = get_secret('GOOGLE_OAUTH_CERTS_URI') or "https://www.googleapis.com/oauth2/v1/certs"
            client_secret = get_secret('GOOGLE_OAUTH_CLIENT_SECRET')
            redirect_uris = get_secret('GOOGLE_OAUTH_REDIRECT_URIS')
            js_origins = get_secret('GOOGLE_OAUTH_JS_ORIGINS')
            
            # Parse redirect URIs and JS origins if they're strings
            if isinstance(redirect_uris, str):
                redirect_uris = [uri.strip() for uri in redirect_uris.split(',')]
            if isinstance(js_origins, str):
                js_origins = [origin.strip() for origin in js_origins.split(',')]
            
            # Build the OAuth client configuration
            if client_id and client_secret:
                oauth_config = {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": auth_uri,
                    "token_uri": token_uri,
                    "auth_provider_x509_cert_url": certs_uri,
                }
                
                # Add optional fields
                if project_id:
                    oauth_config["project_id"] = project_id
                if redirect_uris:
                    oauth_config["redirect_uris"] = redirect_uris
                if js_origins:
                    oauth_config["javascript_origins"] = js_origins
                
                # Wrap in the expected structure
                creds_blob = {"web": oauth_config}
                logger.info("Successfully constructed OAuth configuration from individual secrets")
            else:
                logger.error("Missing required OAuth components: client_id and/or client_secret")
                return
                
        except Exception as e:
            logger.exception(f"Failed to build credentials from individual secrets: {e}")
            return
    
    # Validate the credentials structure
    if not creds_blob:
        logger.error("No valid credentials configuration found")
        return
    
    # Ensure it has the expected structure
    if "web" not in creds_blob and "installed" not in creds_blob:
        logger.error("Invalid credentials structure - missing 'web' or 'installed' key")
        return
    
    # Write to file
    try:
        os.makedirs(os.path.dirname(CREDENTIALS_FILE), exist_ok=True)
        with open(CREDENTIALS_FILE, "w", encoding="utf-8") as f:
            json.dump(creds_blob, f, indent=2)
        logger.info(f"credentials.json created successfully at {CREDENTIALS_FILE}")
    except Exception as e:
        logger.exception(f"Failed to write credentials.json to disk: {e}")
        return

# Initialize credentials file on module load
_ensure_credentials_file()

# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def get_user_token_file(email: str) -> str:
    """Generate a deterministic filename for a user's cached token."""
    return f"token_{hashlib.md5(email.encode()).hexdigest()}.json"


# ---------------------------------------------------------------------------
# Google-Calendar service
# ---------------------------------------------------------------------------

def get_calendar_service(user_email: str):
    """Return an authenticated Calendar service for the given user."""
    try:
        cleanup_expired_credentials()

        creds_json = load_user_credentials(user_email)
        creds = None
        if creds_json:
            import google.oauth2.credentials  # local import to avoid heavy cost at module load

            creds = google.oauth2.credentials.Credentials.from_authorized_user_info(
                json.loads(creds_json), SCOPES
            )
            if creds and creds.valid:
                return build("calendar", "v3", credentials=creds)

            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(google.auth.transport.requests.Request())
                    store_user_credentials(user_email, creds)
                    return build("calendar", "v3", credentials=creds)
                except Exception as refresh_error:
                    logger.error("Token refresh failed for %s: %s", user_email, refresh_error)
                    delete_user_credentials(user_email)

        raise Exception(f"Authentication required for {user_email}")
    except Exception:
        logger.exception("Google Calendar setup failed for %s", user_email)
        raise


# ---------------------------------------------------------------------------
# OAuth helpers
# ---------------------------------------------------------------------------

def generate_auth_url(user_email: str, session_id: str = None) -> str:
    """Kick off OAuth and return the consent URL with session tracking."""
    if not os.path.exists(CREDENTIALS_FILE):
        _ensure_credentials_file()
    if not os.path.exists(CREDENTIALS_FILE):
        # Fallback: credentials.json may live in agent/credentials.json (Streamlit misplacement)
        alt_path = _ROOT_DIR / "agent" / "credentials.json"
        if os.path.exists(alt_path):
            logger.warning("Using fallback credentials file at %s", alt_path)
            cred_path = alt_path
        else:
            raise FileNotFoundError("credentials.json not found – cannot initiate OAuth flow")
    else:
        cred_path = CREDENTIALS_FILE

    # Generate session_id if not provided
    if not session_id:
        session_id = uuid.uuid4().hex
        logger.info(f"Generated new session_id for OAuth: {session_id}")

    flow = InstalledAppFlow.from_client_secrets_file(str(cred_path), SCOPES)
    flow.redirect_uri = REDIRECT_URI
    
    # Use session_id as state parameter for tracking
    auth_url, _ = flow.authorization_url(
        prompt="consent", 
        access_type="offline", 
        include_granted_scopes="true",
        state=session_id  # Critical: pass session_id as state for callback tracking
    )

    user_credentials[user_email] = {
        "flow": flow,
        "auth_url": auth_url,
        "status": "pending_auth",
        "created_at": datetime.now(),
        "session_id": session_id,  # Store session_id for callback matching
    }
    
    logger.info("Generated auth URL for %s with session_id: %s", user_email, session_id)
    return auth_url


def complete_oauth_flow(user_email: str, auth_code: str) -> bool:
    try:
        if user_email not in user_credentials:
            raise Exception("No pending authorization for this user")

        flow = user_credentials[user_email]["flow"]
        flow.fetch_token(code=auth_code)
        store_user_credentials(user_email, flow.credentials)
        user_credentials[user_email]["status"] = "authorized"
        return True
    except Exception:
        logger.exception("OAuth completion failed for %s", user_email)
        return False


def check_auth_completion(user_email: str) -> bool:
    """Helper for periodic polling."""
    return user_credentials.get(user_email, {}).get("status") == "completed"


# ---------------------------------------------------------------------------
# Global Appointments Database
# ---------------------------------------------------------------------------

def init_global_appointments_db():
    """Initialize global appointments table"""
    try:
        conn = sqlite3.connect("user_data.db")
        c = conn.cursor()
        
        # Create global appointments table
        c.execute("""
            CREATE TABLE IF NOT EXISTS global_appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                date TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                title TEXT DEFAULT 'Appointment',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for faster queries
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_global_appointments_date_time 
            ON global_appointments(date, start_time, end_time)
        """)
        
        conn.commit()
        conn.close()
        logger.info("Global appointments database initialized")
    except Exception as e:
        logger.exception(f"Error initializing global appointments DB: {e}")

def check_global_slot_availability(date: str, start_time: str, end_time: str) -> bool:
    """Check if a time slot is available globally"""
    try:
        conn = sqlite3.connect("user_data.db")
        c = conn.cursor()
        
        # Check for any overlapping appointments
        c.execute("""
            SELECT COUNT(*) FROM global_appointments 
            WHERE date = ? AND (
                (start_time <= ? AND end_time > ?) OR
                (start_time < ? AND end_time >= ?) OR
                (start_time >= ? AND end_time <= ?)
            )
        """, (date, start_time, start_time, end_time, end_time, start_time, end_time))
        
        count = c.fetchone()[0]
        conn.close()
        
        return count == 0
    except Exception as e:
        logger.exception(f"Error checking slot availability: {e}")
        return False

def get_available_slots_for_date(date: str, preferred_start: str = "09:00 AM", preferred_end: str = "06:00 PM") -> List[str]:
    """Get all available 30-minute slots for a given date"""
    try:
        # Convert times to datetime objects for calculation
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        start_dt = datetime.strptime(f"{date} {preferred_start}", "%Y-%m-%d %I:%M %p")
        end_dt = datetime.strptime(f"{date} {preferred_end}", "%Y-%m-%d %I:%M %p")
        
        available_slots = []
        current_time = start_dt
        
        while current_time + timedelta(minutes=30) <= end_dt:
            slot_start = current_time.strftime("%I:%M %p")
            slot_end = (current_time + timedelta(minutes=30)).strftime("%I:%M %p")
            
            if check_global_slot_availability(date, slot_start, slot_end):
                available_slots.append(f"{slot_start} - {slot_end}")
            
            current_time += timedelta(minutes=30)
        
        return available_slots
    except Exception as e:
        logger.exception(f"Error getting available slots: {e}")
        return []

def book_global_appointment(user_email: str, date: str, start_time: str, end_time: str, title: str = "Appointment") -> Dict[str, Any]:
    """Book an appointment globally"""
    try:
        # Double-check availability before booking
        if not check_global_slot_availability(date, start_time, end_time):
            return {
                "status": "unavailable",
                "error": "Time slot is no longer available"
            }
        
        conn = sqlite3.connect("user_data.db")
        c = conn.cursor()
        
        c.execute("""
            INSERT INTO global_appointments (user_email, date, start_time, end_time, title)
            VALUES (?, ?, ?, ?, ?)
        """, (user_email, date, start_time, end_time, title))
        
        appointment_id = c.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Global appointment booked: {appointment_id} for {user_email}")
        return {
            "status": "confirmed",
            "appointment_id": appointment_id,
            "slot": f"{start_time} - {end_time}",
            "date": date
        }
    except Exception as e:
        logger.exception(f"Error booking global appointment: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

def get_all_global_appointments(user_email: str) -> List[Dict[str, Any]]:
    """Get all appointments for a user from global database"""
    try:
        conn = sqlite3.connect("user_data.db")
        c = conn.cursor()
        
        c.execute("""
            SELECT id, date, start_time, end_time, title, created_at
            FROM global_appointments 
            WHERE user_email = ?
            ORDER BY date ASC, start_time ASC
        """, (user_email,))
        
        appointments = []
        for row in c.fetchall():
            appointments.append({
                "id": row[0],
                "date": row[1],
                "start_time": row[2],
                "end_time": row[3],
                "title": row[4],
                "created_at": row[5]
            })
        
        conn.close()
        return appointments
    except Exception as e:
        logger.exception(f"Error getting user appointments: {e}")
        return []

def get_conflicting_appointments(date: str, start_time: str, end_time: str) -> List[Dict[str, Any]]:
    """Get appointments that conflict with the requested time slot"""
    try:
        conn = sqlite3.connect("user_data.db")
        c = conn.cursor()
        
        c.execute("""
            SELECT user_email, start_time, end_time, title FROM global_appointments 
            WHERE date = ? AND (
                (start_time <= ? AND end_time > ?) OR
                (start_time < ? AND end_time >= ?) OR
                (start_time >= ? AND end_time <= ?)
            )
        """, (date, start_time, start_time, end_time, end_time, start_time, end_time))
        
        conflicts = []
        for row in c.fetchall():
            conflicts.append({
                "user_email": row[0],
                "start_time": row[1],
                "end_time": row[2],
                "title": row[3]
            })
        
        conn.close()
        return conflicts
    except Exception as e:
        logger.exception(f"Error getting conflicting appointments: {e}")
        return []

# ---------------------------------------------------------------------------
# Calendar utilities
# ---------------------------------------------------------------------------
def parse_time_range(date_str: str, time_range: Dict[str, str]):
    """Fixed time parsing with proper timezone handling"""
    try:
        # Parse the date string and time range
        start_dt = datetime.strptime(f"{date_str} {time_range['start']}", "%Y-%m-%d %I:%M %p")
        end_dt = datetime.strptime(f"{date_str} {time_range['end']}", "%Y-%m-%d %I:%M %p")
        
        # Don't convert to UTC here - keep in local time for comparison
        return start_dt, end_dt
    except Exception as e:
        logger.exception("Time parsing error: %s", e)
        return None, None

def get_user_timezone(service):
    """Get user's timezone from Google Calendar settings"""
    try:
        settings = service.settings().get(setting="timezone").execute()
        timezone_str = settings.get("value", "Asia/Kolkata")
        return timezone_str
    except Exception as e:
        logger.warning("Could not get user timezone: %s", e)
        return "Asia/Kolkata"

def check_availability(state: Dict[str, Any]) -> List[str]:
    """Fixed availability checking with proper timezone handling"""
    try:
        if not state.get("user_email"):
            return ["User email required"]
        if not state.get("date") or not state.get("time_range"):
            return ["Missing date or time information"]

        service = get_calendar_service(state["user_email"])
        user_timezone = get_user_timezone(service)
        
        # Parse the requested time range
        start_dt, end_dt = parse_time_range(state["date"], state["time_range"])
        if not start_dt or not end_dt:
            return ["Invalid date/time format"]

        # Create timezone-aware datetime objects
        tz = pytz.timezone(user_timezone)
        start_dt_tz = tz.localize(start_dt)
        end_dt_tz = tz.localize(end_dt)
        
        # Query calendar for the entire day
        day_start = start_dt_tz.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = start_dt_tz.replace(hour=23, minute=59, second=59, microsecond=0)
        
        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=day_start.isoformat(),
                timeMax=day_end.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = events_result.get("items", [])

        available: List[str] = []
        current_time = start_dt_tz
        
        while current_time < end_dt_tz:
            slot_end = current_time + timedelta(minutes=30)
            if slot_end > end_dt_tz:
                break
                
            is_available = True
            for event in events:
                if 'dateTime' in event['start']:
                    event_start = parser.isoparse(event['start']['dateTime'])
                    event_end = parser.isoparse(event['end']['dateTime'])
                    
                    # Convert to same timezone for comparison
                    if event_start.tzinfo is None:
                        event_start = tz.localize(event_start)
                    if event_end.tzinfo is None:
                        event_end = tz.localize(event_end)
                    
                    # Check for overlap
                    if current_time < event_end and slot_end > event_start:
                        is_available = False
                        break
            
            if is_available:
                # Format time in 12-hour format for display
                available.append(current_time.strftime('%I:%M %p'))
            
            current_time += timedelta(minutes=30)

        if not available:
            return [f"No available slots found in the requested time range ({start_dt_tz.strftime('%I:%M %p')} - {end_dt_tz.strftime('%I:%M %p')}) on {state['date']}. Please try a different time or date."]
        
        return available
        
    except Exception as e:
        msg = str(e)
        if "Authorization required" in msg or "Authentication required" in msg:
            return [f"AUTH_REQUIRED: {msg}"]
        logger.exception("Availability check failed")
        return [f"Error checking availability: {msg}"]

def book_appointment(state: Dict[str, Any], slot: str):
    """Fixed appointment booking with proper timezone handling"""
    try:
        if not state.get("user_email"):
            return {"status": "error", "slot": slot, "error": "User email required"}
        if not state.get("date"):
            return {"status": "error", "slot": slot, "error": "Date is required"}

        service = get_calendar_service(state["user_email"])
        user_timezone = get_user_timezone(service)
        
        # Parse the date and time
        try:
            if isinstance(state["date"], str):
                if len(state["date"]) == 10 and "-" in state["date"]:
                    date_str = state["date"]
                else:
                    # Handle natural language dates
                    parsed_date = parser.parse(state["date"])
                    date_str = parsed_date.strftime("%Y-%m-%d")
            else:
                date_str = str(state["date"])
            
            # Parse the time slot
            start_time = datetime.strptime(f"{date_str} {slot}", "%Y-%m-%d %I:%M %p")
            end_time = start_time + timedelta(minutes=30)
            
            # Create timezone-aware datetime objects
            tz = pytz.timezone(user_timezone)
            start_time_tz = tz.localize(start_time)
            end_time_tz = tz.localize(end_time)
            
            logger.info(f"Booking appointment: {start_time_tz} to {end_time_tz} ({user_timezone})")
            
        except ValueError as e:
            return {"status": "error", "slot": slot, "error": f"Invalid date/time format: {e}"}

        # Create the event
        event = {
            "summary": "Appointment",
            "description": f"Booked via AI Assistant for {state['user_email']}",
            "start": {
                "dateTime": start_time_tz.isoformat(),
                "timeZone": user_timezone
            },
            "end": {
                "dateTime": end_time_tz.isoformat(), 
                "timeZone": user_timezone
            },
        }
        
        # Create the event in calendar
        created = service.events().insert(calendarId="primary", body=event).execute()
        
        logger.info(f"Successfully created event: {created.get('id')} at {start_time_tz}")
        
        return {
            "status": "confirmed", 
            "slot": slot, 
            "event_id": created.get("id"),
            "actual_time": start_time_tz.strftime("%Y-%m-%d %I:%M %p %Z")
        }
        
    except Exception as e:
        logger.exception("Booking failed")
        return {"status": "error", "slot": slot, "error": str(e)}

def get_user_appointments(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fixed appointment retrieval with proper timezone handling"""
    try:
        if not state.get("user_email"):
            return []
            
        service = get_calendar_service(state["user_email"])
        user_timezone = get_user_timezone(service)
        tz = pytz.timezone(user_timezone)
        
        # Get current time in user's timezone
        now = datetime.now(tz)
        time_min = now.isoformat()
        time_max = (now + timedelta(days=30)).isoformat()
        
        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=time_min,
                timeMax=time_max,
                maxResults=50,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = events_result.get("items", [])

        appointments: List[Dict[str, Any]] = []
        for event in events:
            # Check if event summary contains "appointment" (case insensitive)
            summary = event.get('summary', '').lower()
            if 'appointment' in summary:
                # Handle both dateTime and date formats
                if 'dateTime' in event['start']:
                    event_start = parser.isoparse(event['start']['dateTime'])
                    event_end = parser.isoparse(event['end']['dateTime'])
                    
                    # Convert to user's timezone for display
                    if event_start.tzinfo is None:
                        event_start = tz.localize(event_start)
                    else:
                        event_start = event_start.astimezone(tz)
                    
                    if event_end.tzinfo is None:
                        event_end = tz.localize(event_end)
                    else:
                        event_end = event_end.astimezone(tz)
                    
                    appointments.append({
                        'title': event.get('summary', 'Appointment'),
                        'date': event_start.strftime('%Y-%m-%d'),
                        'start_time': event_start.strftime('%I:%M %p'),
                        'end_time': event_end.strftime('%I:%M %p'),
                        'description': event.get('description', ''),
                        'datetime': event_start,
                        'timezone': user_timezone
                    })
                else:
                    # All-day event
                    date_str = event['start']['date']
                    appointments.append({
                        'title': event.get('summary', 'Appointment'),
                        'date': date_str,
                        'start_time': 'All day',
                        'end_time': 'All day',
                        'description': event.get('description', ''),
                        'datetime': datetime.strptime(date_str, '%Y-%m-%d'),
                        'timezone': user_timezone
                    })
        
        # Sort by datetime
        appointments.sort(key=lambda x: x['datetime'])
        return appointments
        
    except Exception as e:
        logger.exception("Failed to get appointments for %s", state.get("user_email"))
        return []
