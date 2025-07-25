import json
import logging
import threading
import time
import uuid
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
import sqlite3
from dotenv import load_dotenv
import streamlit as st

from agent.models.database import (
    cleanup_expired_credentials,
    delete_user_credentials,
    load_conversation_state,
    load_user_credentials,
    reset_database,
    save_conversation_state,
    store_user_credentials,
    check_database_health,
    user_credentials,
)
from agent.models.calendar import (
    book_appointment,
    check_availability,
    complete_oauth_flow,
    generate_auth_url,
    get_calendar_service,
    get_user_appointments,
    get_user_timezone,
    parse_time_range,
    init_global_appointments_db,
    check_global_slot_availability,
    get_available_slots_for_date,
    book_global_appointment,
    get_all_global_appointments,
    get_conflicting_appointments,
    CREDENTIALS_FILE as _CRED_PATH
)


init_global_appointments_db() 

import google.generativeai as genai
from dateutil import parser
from groq import Groq

# ---------------------------------------------------------------------------
# Logger first so it is available for any early warnings
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment / secret loading
# ---------------------------------------------------------------------------

load_dotenv()  # Loads variables from a .env file into process env, if present

# ---------------------------------------------------------------------------
# Unified secret getter (Streamlit native or env fallback)
# ---------------------------------------------------------------------------

# Production environment detection
IS_PRODUCTION = os.getenv("ENV", "").lower() == "production"
FRONTEND_URL = "https://ai-booking-agent-efd.streamlit.app" if IS_PRODUCTION else "http://localhost:8501"

logger.info(f"Environment: {'PRODUCTION' if IS_PRODUCTION else 'DEVELOPMENT'}")
logger.info(f"Frontend URL: {FRONTEND_URL}")


def get_secret(key: str, default: str = "") -> str:
    """Fetch secret: prefer OS env, then Streamlit secrets (for UI runs)."""
    # 1) Environment variable (preferred for Railway)
    env_val = os.getenv(key)
    if env_val not in (None, ""):
        return env_val

    # 2) Streamlit secrets fallback (for Streamlit Cloud/local)
    try:
        import streamlit as _st  # local import; may not exist in backend container
        if key in _st.secrets:
            return _st.secrets[key]
        if "default" in _st.secrets and key in _st.secrets["default"]:
            return _st.secrets["default"][key]
    except ModuleNotFoundError:
        pass
    except Exception as e:
        logger.warning("Streamlit secrets access error for %s: %s", key, e)

    if key == "GOOGLE_CREDENTIALS_JSON":
        logger.error("%s not found in environment or Streamlit secrets", key)

    return default

# ---------------------------------------------------------------------------
# Ensure credentials.json exists BEFORE OAuth calls (redundant fallback)
# ---------------------------------------------------------------------------

_creds_blob = get_secret("GOOGLE_CREDENTIALS_JSON")
logger.info("GOOGLE_CREDENTIALS_JSON length: %s", len(_creds_blob or ""))
logger.info("CREDENTIALS_FILE path: %s", _CRED_PATH)

if _creds_blob:
    try:
        # Parse JSON to validate format
        creds_data = json.loads(_creds_blob)
        if not isinstance(creds_data, dict) or "web" not in creds_data:
            raise ValueError("Invalid credentials format - missing 'web' key")
        
        if not os.path.exists(_CRED_PATH):
            os.makedirs(os.path.dirname(_CRED_PATH), exist_ok=True)
            with open(_CRED_PATH, "w", encoding="utf-8") as _f:
                json.dump(creds_data, _f, indent=2)
            logger.info("credentials.json written successfully at %s", _CRED_PATH)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse GOOGLE_CREDENTIALS_JSON: %s", e)
    except Exception as e:
        logger.exception("Failed to write credentials.json: %s", e)
else:
    logger.error("GOOGLE_CREDENTIALS_JSON environment variable is not set")

# Gemini and Groq configuration pulled from environment variables
GROQ_API_KEY = get_secret("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY is not set")

GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY is not set - Gemini calls will fail")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

# ---------------------------------------------------------------------------
# Browser-specific cookie session settings
# ---------------------------------------------------------------------------
SESSION_COOKIE_NAME = "booking_session"
SESSION_COOKIE_MAX_AGE = 30 * 24 * 60 * 60  # 30 days


def is_secure_cookie() -> bool:
    """Return True if we should mark cookies as Secure (i.e., only over HTTPS)."""
    import os

    return os.getenv("ENV", "").lower() == "production"


@app.middleware("http")
async def session_cookie_middleware(request: Request, call_next):
    """Production-ready session cookie middleware"""
    
    # CRITICAL: Read cookie from request FIRST
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    
    # Debug logging to see what's happening
    logger.debug(f"Cookie from request: {session_id}")
    logger.debug(f"All cookies: {dict(request.cookies)}")
    
    # Only generate new session ID if absolutely no cookie exists
    if not session_id:
        session_id = uuid.uuid4().hex
        logger.info(f"Generated NEW session ID (no cookie found): {session_id}")
    else:
        logger.debug(f"Using EXISTING session ID from cookie: {session_id}")
    
    request.state.session_id = session_id
    response = await call_next(request)
    
    # CRITICAL: Production cookie configuration
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=SESSION_COOKIE_MAX_AGE,
        httponly=False,  # MUST be False for your frontend to read it
        samesite="none",  # Critical for cross-origin requests
        secure=True,     # MUST be True in production (HTTPS)
        path="/",
        domain=None      # Let browser handle domain automatically
    )
    
    logger.debug(f"Set session cookie in response: {session_id}")
    return response


def resolve_browser_session_id(request: Request, provided_id: Optional[str] = None) -> Optional[str]:
    """Fixed session resolution with proper cookie priority"""
    
    # Priority 1: Cookie from browser (most reliable)
    cookie_session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if cookie_session_id:
        logger.debug(f"Found session in cookie: {cookie_session_id}")
        return cookie_session_id
    
    # Priority 2: Request state (set by middleware)
    if hasattr(request.state, "session_id"):
        state_session_id = request.state.session_id
        if state_session_id:
            logger.debug(f"Found session in request state: {state_session_id}")
            return state_session_id
    
    # Priority 3: Provided parameter (fallback)
    if provided_id:
        logger.debug(f"Using provided session ID: {provided_id}")
        return provided_id
    
    logger.warning("No session ID found in any source")
    return None


def call_llama_groq(user_input: str) -> dict:
    client = Groq(api_key=GROQ_API_KEY)
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            model="llama-3.3-70b-versatile",
            stream=False,
        )
        content = chat_completion.choices[0].message.content
        return {"intent": "unknown", "response": content}
    except Exception as e:
        return {"intent": "unknown", "error": f"Llama fallback failed: {e}"}

def fallback_parse_input(user_input: str, today: str) -> dict:
    """Simple fallback parser when Gemini fails"""
    user_input_lower = user_input.lower()
    
    # Basic intent detection
    if any(word in user_input_lower for word in ["show", "my appointments", "meetings", "schedule"]):
        intent = "show_appointments"
    elif any(word in user_input_lower for word in ["book", "schedule", "appointment"]):
        intent = "book_appointment" 
    elif any(word in user_input_lower for word in ["check", "availability", "free"]):
        intent = "check_availability"
    else:
        intent = "unknown"
    
    # Basic date parsing
    date = None
    if "today" in user_input_lower:
        date = today
    elif "tomorrow" in user_input_lower:
        tomorrow = datetime.now() + timedelta(days=1)
        date = tomorrow.strftime("%Y-%m-%d")
    # Basic time parsing
    specific_time = None
    import re
    # Look for time patterns like "11 pm", "11:00 pm", "23:00"
    time_patterns = [
        r'(\d{1,2}):?(\d{0,2})\s*(pm|am)',
        r'(\d{1,2}):(\d{2})'
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, user_input_lower)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            
            if len(match.groups()) >= 3 and match.group(3):  # Has AM/PM
                ampm = match.group(3).upper()
                specific_time = f"{hour:02d}:{minute:02d} {ampm}"
            else:  # 24-hour format
                if hour > 12:
                    ampm = "PM"
                    hour = hour - 12 if hour > 12 else hour
                else:
                    ampm = "AM" if hour < 12 else "PM"
                specific_time = f"{hour:02d}:{minute:02d} {ampm}"
            break
    
    return {
        "intent": intent,
        "date": date,
        "time_range": None,
        "specific_time": specific_time,
        "email": None
    }

def call_gemini(user_input: str, context: Dict[str, Any] | None = None, retries: int = 3):
    logger.info("Calling Gemini API with input: %s", user_input)
    
    # Get current date and time in user's timezone
    from datetime import datetime
    import pytz
    # Assume Indian timezone for now, but this should come from user settings
    user_tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(user_tz)
    today = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%I:%M %p")
    
    context_str = (
        f"Previous conversation context: Intent={context['last_intent']}, Date={context.get('date')}, Time={context.get('time_range')}"
        if context and context.get("last_intent")
        else ""
    )

    appointment_intents = [
        "show my appointments",
        "my meetings", 
        "my calendar",
        "my schedule",
        "upcoming appointments",
        "what meetings do i have",
        "show appointments",
        "view my schedule"
    ]

    prompt = f"""
    You are an appointment booking assistant. Analyze the user input and return a JSON object.
    
    CURRENT CONTEXT:
    - Current date: {today}
    - Current time: {current_time}
    - User timezone: Asia/Kolkata
    {context_str}
    
    IMPORTANT DATE/TIME PARSING RULES:
    1. "today" = {today}
    2. "tomorrow" = {(now + timedelta(days=1)).strftime("%Y-%m-%d")}
    3. When user says "11 pm" or "11:00 pm", convert to "11:00 PM" format
    4. When user says "23:00", convert to "11:00 PM" format
    5. DO NOT convert times to UTC - keep them in user's local timezone
    6. For booking appointments, if only a specific time is given (like "11 pm"), create a 30-minute slot
    
    User input: "{user_input}"
    
    Appointment-related intents: {appointment_intents}
    If the user's message matches any of these, set "intent": "show_appointments".
    
    Return JSON with:
    - "intent": One of ["show_appointments", "book_appointment", "check_availability", "greeting", "unknown"]
    - "date": Date in YYYY-MM-DD format (e.g., "{today}" for today) or null
    - "time_range": {{"start": "HH:MM AM/PM", "end": "HH:MM AM/PM"}} or null
    - "specific_time": Single time slot in 12-hour format (e.g., "11:00 PM") or null
    - "email": Email address if mentioned or null
    
    EXAMPLES:
    Input: "book appointment today 11 pm"
    Output: {{
        "intent": "book_appointment",
        "date": "{today}",
        "time_range": null,
        "specific_time": "11:00 PM",
        "email": null
    }}
    
    Input: "book meeting tomorrow 2-4 PM"  
    Output: {{
        "intent": "book_appointment",
        "date": "{(now + timedelta(days=1)).strftime("%Y-%m-%d")}",
        "time_range": {{"start": "2:00 PM", "end": "4:00 PM"}},
        "specific_time": null,
        "email": null
    }}
    
    Input: "show my appointments"
    Output: {{
        "intent": "show_appointments", 
        "date": null,
        "time_range": null,
        "specific_time": null,
        "email": null
    }}
    """

    for model_name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            for attempt in range(retries):
                try:
                    response = model.generate_content(prompt)
                    if not response or not response.text:
                        continue
                    text = response.text.strip()
                    if text.startswith("```json"):
                        text = text.replace("```json", "").replace("```", "").strip()
                    
                    result = json.loads(text)
                    logger.info(f"Gemini parsed result: {result}")
                    return result
                    
                except Exception as e:
                    if "429" in str(e) and attempt < retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    logger.warning("Gemini error (%s): %s", model_name, e)
                    break
        except Exception as e:
            logger.warning("Model %s unavailable: %s", model_name, e)
    
    # Fallback to simple parsing if Gemini fails
    logger.warning("Gemini failed, using fallback parsing")
    return fallback_parse_input(user_input, today)

class BookingRequest(BaseModel):
    user_input: str
    conversation_id: str
    user_email: Optional[str] = None

from typing_extensions import TypedDict

class AgentState(TypedDict, total=False):
    user_input: str
    conversation_id: str
    user_email: str
    intent: str
    date: Optional[str]
    time_range: Optional[Dict[str, str]]
    specific_time: Optional[str]
    available_slots: List[str]
    confirmed_slot: str
    messages: List[Dict[str, str]]
    last_intent: str
    needs_more_info: bool
    auth_required: bool
    auth_url: str

def email_collection_node(state: Dict[str, Any]):
    """Check if user email is provided and handle authentication - ALWAYS runs first"""
    state['messages'] = state.get('messages', [])
    # ALWAYS check for authentication, regardless of intent
    if not state.get('user_email'):
        # Try to extract email from user input
        import re
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.findall(email_pattern, state['user_input'])
        
        if email_matches:
            state['user_email'] = email_matches[0]
            logger.info(f"Email extracted: {state['user_email']}")
        else:
            # Try Gemini to extract email
            gemini_result = call_gemini(state['user_input'])
            if gemini_result.get('email'):
                state['user_email'] = gemini_result['email']
                logger.info(f"Email extracted via Gemini: {state['user_email']}")
            else:
                # Ask for email - even for simple greetings
                greeting_response = "Hi! I'm your AI appointment booking assistant. "
                if "hi" in state['user_input'].lower() or "hello" in state['user_input'].lower():
                    greeting_response += "Nice to meet you! "
                greeting_response += "To access your calendar and help with appointments, please provide your email address."
                
                state['messages'].append({
                    'role': 'assistant',
                    'content': greeting_response
                })
                state['needs_more_info'] = True
                return state
    
    # Check if user has valid cached credentials
    if state.get('user_email'):
        creds_json = load_user_credentials(state['user_email'])
        
        if creds_json:
            # User has valid cached credentials
            try:
                service = get_calendar_service(state['user_email'])
                state['needs_more_info'] = False
                state['auth_required'] = False
                logger.info(f"Using cached credentials for {state['user_email']}")
                return state
            except Exception as e:
                logger.error(f"Cached credentials failed for {state['user_email']}: {e}")
                delete_user_credentials(state['user_email'])
                state['auth_required'] = True
                state['needs_more_info'] = False
                # Generate auth URL immediately
                try:
                    state['auth_url'] = generate_auth_url(state['user_email'])
                except Exception as auth_e:
                    logger.error(f"Failed to generate auth URL: {auth_e}")
        else:
            state['auth_required'] = True
            state['needs_more_info'] = False
            # Generate auth URL immediately
            try:
                state['auth_url'] = generate_auth_url(state['user_email'])
            except Exception as auth_e:
                logger.error(f"Failed to generate auth URL: {auth_e}")
            logger.info(f"No cached credentials for {state['user_email']}, requiring auth")
    
    return state

def intent_detection_node(state: Dict[str, Any]):
    if state.get("needs_more_info") or state.get("auth_required"):
        return state

    ctx = {
        "last_intent": state.get("last_intent", ""),
        "date": state.get("date"),
        "time_range": state.get("time_range"),
    }
    gemini = call_gemini(state["user_input"], ctx)
    state["intent"] = gemini.get("intent", "unknown")
    new_date = gemini.get("date")
    new_time_range = gemini.get("time_range")

    if new_date is not None:
        state["date"] = new_date
    if new_time_range is not None:
        state["time_range"] = new_time_range
        # Reset previously suggested slots since user provided new time
        state["available_slots"] = []
    new_specific = gemini.get("specific_time")
    state["specific_time"] = new_specific
    if new_specific is not None:
        # Clear previous range so validation converts new specific time
        state["time_range"] = None
        state["available_slots"] = []
    state["last_intent"] = state["intent"]
    return state


def validation_node(state: Dict[str, Any]):
    """Enhanced validation that handles greetings and authentication properly"""
    # If we're waiting for OAuth or need more info (like email), skip validation
    if state.get("auth_required") or state.get("needs_more_info"):
        return state

    state["messages"] = state.get("messages", [])
    user_input_lower = state.get("user_input", "").lower()

    # Handle pure greetings when user is authenticated (avoid false-positives like "hey I want to book")
    import re
    greeting_only_pattern = r"^(hi|hello|hey|good\s+(morning|afternoon|evening))\W*$"
    if re.match(greeting_only_pattern, user_input_lower.strip()):
        state["messages"].append({
            "role": "assistant",
            "content": f"Hello! I'm your appointment assistant. I can help you:\n\n• Book new appointments\n• Check your availability\n• Show your upcoming meetings\n\nWhat would you like to do today?"
        })
        state["needs_more_info"] = True  # Wait for their next input
        return state

    if state.get("intent") == "show_appointments":
        state["needs_more_info"] = False
    elif state.get("intent") in ["book_appointment", "check_availability"]:
        if not state.get("date"):
            state["messages"].append({
                "role": "assistant",
                "content": "I'd be happy to help! What date would you like? You can say something like 'tomorrow', 'Friday', or a specific date.",
            })
            state["needs_more_info"] = True
        elif not state.get("time_range") and not state.get("specific_time"):
            state["messages"].append({
                "role": "assistant",
                "content": f"Great! For {state['date']}, what time works for you? You can specify a range like '2-4 PM' or a specific time like '3 PM'.",
            })
            state["needs_more_info"] = True
        else:
            # Convert specific_time to time_range if needed (existing logic)
            if state.get('specific_time') and not state.get('time_range'):
                try:
                    time_str = state['specific_time']
                    
                    # Handle 24-hour format
                    if ':' in time_str and 'AM' not in time_str.upper() and 'PM' not in time_str.upper():
                        # Convert 24-hour to 12-hour format
                        try:
                            dt = datetime.strptime(time_str, '%H:%M')
                            time_str = dt.strftime('%I:%M %p')
                        except ValueError:
                            raise ValueError("Invalid 24-hour time format")
                    elif ':' not in time_str and ('AM' in time_str.upper() or 'PM' in time_str.upper()):
                        # Handle formats like "8 PM"
                        time_str = time_str.replace(' ', ':00 ')
                    
                    specific_dt = datetime.strptime(time_str, '%I:%M %p')
                    end_dt = specific_dt + timedelta(minutes=30)
                    state['time_range'] = {
                        'start': specific_dt.strftime('%I:%M %p'),
                        'end': end_dt.strftime('%I:%M %p')
                    }
                    logger.info("Converted specific time %s to range %s", time_str, state['time_range'])
                except Exception as e:
                    logger.error("Time conversion error: %s", e)
                    state["messages"].append({
                        "role": "assistant",
                        "content": "Sorry, I couldn't understand the time format. Please specify like '8:00 PM', '8 PM', or '20:00'.",
                    })
                    state["needs_more_info"] = True
                    return state
            state["needs_more_info"] = False
    else:
        # Unknown intent - provide helpful guidance
        state["messages"].append({
            "role": "assistant",
            "content": "I can help you with appointments! Try saying something like:\n\n• 'Book a meeting tomorrow at 3 PM'\n• 'Check my availability Friday afternoon'\n• 'Show my upcoming appointments'\n\nWhat would you like to do?",
        })
        state["needs_more_info"] = True
    return state

def availability_node(state: Dict[str, Any]):
    """Enhanced availability node with global slot checking and negotiation"""
    if state.get("needs_more_info") or state.get("auth_required"):
        return state
    
    state.setdefault("messages", [])
    
    if state.get("intent") in ["book_appointment", "check_availability"] and state.get("date") and state.get("time_range"):
        date = state["date"]
        time_range = state["time_range"]
        start_time = time_range["start"]
        end_time = time_range["end"]
        
        # Check if the specific requested slot is available
        if check_global_slot_availability(date, start_time, end_time):
            state["available_slots"] = [f"{start_time} - {end_time}"]
            state["messages"].append({
                "role": "assistant",
                "content": f"Great! The slot {start_time} - {end_time} on {date} is available. Would you like me to book it for you?",
            })
        else:
            # Slot is not available - show conflicts and alternatives
            conflicts = get_conflicting_appointments(date, start_time, end_time)
            conflict_msg = f"Sorry, the slot {start_time} - {end_time} on {date} is already booked"
            
            if conflicts:
                conflict_msg += f" (booked by other user)"
            
            # Get alternative slots for the same day
            available_alternatives = get_available_slots_for_date(date)
            
            if available_alternatives:
                # Limit to first 5 alternatives for better UX
                alt_slots = available_alternatives[:5]
                state["available_slots"] = alt_slots
                
                alternatives_text = "\n".join([f"• {slot}" for slot in alt_slots])
                state["messages"].append({
                    "role": "assistant", 
                    "content": f"{conflict_msg}.\n\nHere are some available alternatives for {date}:\n\n{alternatives_text}\n\nWhich slot would you prefer? Just say the time (e.g., '2:00 PM - 2:30 PM')."
                })
                state["needs_more_info"] = True  # Wait for user to choose alternative
            else:
                state["messages"].append({
                    "role": "assistant",
                    "content": f"{conflict_msg}.\n\nUnfortunately, there are no other available slots on {date}. Would you like to try a different date?"
                })
                state["needs_more_info"] = True
                state["available_slots"] = []
    
    elif state.get("intent") == "check_availability" and state.get("date"):
        # Just checking availability, not booking
        date = state["date"]
        available_slots = get_available_slots_for_date(date)
        
        if available_slots:
            slots_text = "\n".join([f"• {slot}" for slot in available_slots[:10]])  # Show max 10 slots
            state["messages"].append({
                "role": "assistant",
                "content": f"Here are the available slots for {date}:\n\n{slots_text}\n\nWould you like to book any of these slots?"
            })
            state["available_slots"] = available_slots
        else:
            state["messages"].append({
                "role": "assistant",
                "content": f"Sorry, there are no available slots on {date}. Would you like to check a different date?"
            })
            state["available_slots"] = []
    
    return state

def booking_node(state: Dict[str, Any]):
    """Enhanced booking node using global appointment system"""
    if state.get("needs_more_info") or state.get("auth_required"):
        return state
    
    state.setdefault("messages", [])
    
    if state.get("intent") == "book_appointment":
        user_input_lower = state.get("user_input", "").lower()
        
        # Check if user is selecting from available slots
        if state.get("available_slots"):
            selected_slot = None
            # Try to match user input with available slots
            for slot in state["available_slots"]:
                # Extract time from slot format "HH:MM AM/PM - HH:MM AM/PM"
                slot_times = slot.split(" - ")
                if len(slot_times) == 2:
                    start_time = slot_times[0].strip()
                    end_time = slot_times[1].strip()
                    
                    # Check if user mentioned this time
                    if (start_time.lower() in user_input_lower or 
                        slot.lower() in user_input_lower or
                        start_time.replace(":00", "").lower() in user_input_lower):
                        selected_slot = slot
                        break
            
            # If no specific slot mentioned, use the first available
            if not selected_slot and state["available_slots"]:
                selected_slot = state["available_slots"][0]
            
            if selected_slot:
                # Parse the selected slot
                slot_parts = selected_slot.split(" - ")
                if len(slot_parts) == 2:
                    start_time = slot_parts[0].strip()
                    end_time = slot_parts[1].strip()
                    date = state["date"]
                    user_email = state["user_email"]
                    
                    # Book the appointment globally
                    result = book_global_appointment(user_email, date, start_time, end_time)
                    
                    if result["status"] == "confirmed":
                        # Also create event in user's Google Calendar
                        cal_outcome = book_appointment(
                            {
                                "user_email": user_email,
                                "date": date,
                            },
                            start_time,
                        )

                        if cal_outcome.get("status") == "confirmed":
                            calendar_msg = "and added to your Google Calendar"
                        else:
                            calendar_msg = "but I could not add it to your Google Calendar: " + cal_outcome.get("error", "unknown error")

                        state["messages"].append({
                            "role": "assistant",
                            "content": f"✅ Perfect! Your appointment is booked for {date} from {start_time} to {end_time} {calendar_msg}. Your appointment ID is #{result['appointment_id']}. Is there anything else I can help you with?"
                        })
                        # Clear the slots since booking is complete
                        state["available_slots"] = []
                        state["needs_more_info"] = False
                    elif result["status"] == "unavailable":
                        state["messages"].append({
                            "role": "assistant", 
                            "content": f"Sorry, that slot was just booked by someone else! Let me check for other available slots..."
                        })
                        # Refresh available slots
                        new_slots = get_available_slots_for_date(date)
                        state["available_slots"] = new_slots[:5]
                        if new_slots:
                            alternatives_text = "\n".join([f"• {slot}" for slot in new_slots[:5]])
                            state["messages"].append({
                                "role": "assistant",
                                "content": f"Here are the updated available slots:\n\n{alternatives_text}\n\nWhich one would you like?"
                            })
                        state["needs_more_info"] = True
                    else:
                        state["messages"].append({
                            "role": "assistant",
                            "content": f"Sorry, I encountered an error while booking: {result.get('error', 'Unknown error')}. Please try again."
                        })
                        state["needs_more_info"] = True
                else:
                    state["messages"].append({
                        "role": "assistant",
                        "content": "I couldn't parse the selected time slot. Please specify the time again."
                    })
                    state["needs_more_info"] = True
            else:
                state["messages"].append({
                    "role": "assistant",
                    "content": "I couldn't understand which slot you'd like. Please specify the exact time (e.g., '2:00 PM - 2:30 PM') from the available options."
                })
                state["needs_more_info"] = True
        else:
            # No available slots, ask for different time/date
            state["messages"].append({
                "role": "assistant",
                "content": "I don't have any available slots to book. Would you like to try a different date or time?"
            })
            state["needs_more_info"] = True
    
    return state

def show_appointments_node(state: Dict[str, Any]):
    """Enhanced show appointments using global database"""
    if state.get("needs_more_info") or state.get("auth_required"):
        return state
    
    state.setdefault("messages", [])
    
    if state.get("intent") == "show_appointments" and state.get("user_email"):
        appointments = get_all_global_appointments(state["user_email"])
        
        if appointments:
            # Format appointments for display
            appt_list = []
            for appt in appointments:
                appt_list.append(f"• **{appt['title']}** on {appt['date']} from {appt['start_time']} to {appt['end_time']} (ID: #{appt['id']})")
            
            reply = "Here are your upcoming appointments:\n\n" + "\n".join(appt_list)
            reply += "\n\nWould you like to book another appointment or make any changes?"
        else:
            reply = "You have no upcoming appointments. Would you like to book one?"
        
        state["messages"].append({"role": "assistant", "content": reply})
    
    return state

browser_sessions: Dict[str, Dict[str, any]] = {}

def init_browser_sessions_db():
    """Initialize browser sessions table in database"""
    conn = sqlite3.connect("user_data.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS browser_sessions (
            session_id TEXT PRIMARY KEY,
            user_email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# ---------------------------------------------------------------------------
# Optional Redis-backed session store (provides horizontal scalability). If
# REDIS_URL and SESSION_ENCRYPTION_KEY are configured, the helper below will
# return a fully-functional store; otherwise it is a no-op dummy so legacy
# SQLite behaviour continues to work unchanged.  
# ---------------------------------------------------------------------------

from agent.utils.session_store import get_store as _get_session_store

# Global instance shared by all request handlers
SESSION_STORE = _get_session_store()

def test_redis_connection():
    """Test basic Redis connection and operations"""
    try:
        # Get Redis URL from environment
        redis_url = os.getenv('REDIS_URL')
        if not redis_url:
            print("❌ REDIS_URL not found in environment variables")
            return False
            
        print(f"🔗 Connecting to Redis: {redis_url[:20]}...")
        
        # Create Redis client
        r = redis.from_url(redis_url, decode_responses=True)
        
        # Test basic operations
        print("🧪 Testing basic Redis operations...")
        
        # Test SET and GET
        test_key = "test_key_123"
        test_value = "test_value_456"
        
        result = r.set(test_key, test_value)
        print(f"SET result: {result}")
        
        retrieved = r.get(test_key)
        print(f"GET result: {retrieved}")
        
        if retrieved == test_value:
            print("✅ Basic Redis operations working!")
            
            # Clean up test key
            r.delete(test_key)
            return True
        else:
            print("❌ Basic Redis operations failed!")
            return False
            
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def test_session_store():
    """Test session store operations specifically"""
    try:
        from agent.utils.session_store import get_store
        
        store = get_store()
        print(f"📦 Session store type: {type(store)}")
        
        # Test session operations
        test_session_id = "test_session_12345"
        test_data = {"user_email": "test@example.com", "timestamp": "2024-01-01"}
        
        print("🧪 Testing session store operations...")
        
        # Set session
        store.set_session(test_session_id, test_data)
        print("✅ set_session called")
        
        # Get session
        retrieved_data = store.get_session(test_session_id)
        print(f"📥 Retrieved data: {retrieved_data}")
        
        if retrieved_data == test_data:
            print("✅ Session store working correctly!")
        else:
            print(f"❌ Session store mismatch. Expected: {test_data}, Got: {retrieved_data}")
        
        # Test user sessions
        test_user = "test@example.com"
        store.add_user_session(test_user, test_session_id)
        user_sessions = store.get_user_sessions(test_user)
        print(f"👤 User sessions: {user_sessions}")
        
        # Clean up
        store.delete_session(test_session_id)
        store.remove_user_session(test_user, test_session_id)
        
        return True
        
    except Exception as e:
        print(f"❌ Session store test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_current_session_storage():
    """Debug what's happening in your current session storage"""
    try:
        print("🔍 Debugging current session storage...")
        
        # Test the exact flow from your code
        from agent.utils.session_store import get_store
        
        SESSION_STORE = get_store()
        
        # Simulate store_browser_session function
        session_id = "debug_session_123"
        user_email = "debug@example.com"
        
        print(f"🔧 Storing session: {session_id} -> {user_email}")
        
        try:
            SESSION_STORE.set_session(session_id, {"user_email": user_email})
            print("✅ set_session completed")
            
            SESSION_STORE.add_user_session(user_email, session_id)
            print("✅ add_user_session completed")
            
        except Exception as e:
            print(f"❌ Redis operations failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Try to retrieve
        try:
            data = SESSION_STORE.get_session(session_id)
            print(f"📥 Retrieved session data: {data}")
            
            sessions = SESSION_STORE.get_user_sessions(user_email)
            print(f"📥 Retrieved user sessions: {sessions}")
            
        except Exception as e:
            print(f"❌ Redis retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

def fixed_store_browser_session(session_id: str, user_email: str):
    """Enhanced version of store_browser_session with proper error handling"""
    
    logger.info(f"Storing browser session: {session_id} -> {user_email}")
    
    # Try Redis first with detailed error handling
    try:
        from agent.utils.session_store import get_store
        SESSION_STORE = get_store()
        
        # Check if it's actually a Redis store
        if hasattr(SESSION_STORE, 'redis_client'):
            logger.info("Using Redis session store")
            
            # Test Redis connection first
            SESSION_STORE.redis_client.ping()
            logger.info("Redis ping successful")
            
            # Store session data
            result1 = SESSION_STORE.set_session(session_id, {"user_email": user_email})
            logger.info(f"set_session result: {result1}")
            
            result2 = SESSION_STORE.add_user_session(user_email, session_id)
            logger.info(f"add_user_session result: {result2}")
            
            # Verify storage
            stored_data = SESSION_STORE.get_session(session_id)
            logger.info(f"Verification - stored data: {stored_data}")
            
        else:
            logger.warning("Session store is not Redis-backed, using fallback")
            
    except Exception as e:
        logger.error(f"Redis session store failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # Always use SQLite fallback
    try:
        import sqlite3
        conn = sqlite3.connect("user_data.db")
        c = conn.cursor()
        c.execute(
            """
            INSERT OR REPLACE INTO browser_sessions (session_id, user_email, last_accessed)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (session_id, user_email),
        )
        conn.commit()
        conn.close()
        logger.info("SQLite session storage successful")
        
    except Exception as e:
        logger.error(f"SQLite session storage failed: {e}")

def store_browser_session(session_id: str, user_email: str):
    """Unified session storage in both Redis and memory"""
    logger.info(f"Storing browser session: {session_id} -> {user_email}")
    
    session_data = {
        "user_email": user_email,
        "status": "completed",
        "timestamp": datetime.now().isoformat()
    }
    
    # Store in Redis
    try:
        SESSION_STORE.set_session(session_id, session_data, ttl=60*60*24*30)
        SESSION_STORE.add_user_session(user_email, session_id, ttl=60*60*24*30)
        logger.info("Redis session storage successful")
    except Exception as e:
        logger.error(f"Redis session storage failed: {e}")
    
    # Store in memory for immediate access
    browser_sessions[session_id] = {
        "user_email": user_email,
        "status": "completed",
        "auth_url": "",
    }
    
    logger.info(f"Session stored successfully: {session_id} -> {user_email}")

def get_session_user(session_id: str) -> str:
    """Redis-only session retrieval"""
    try:
        data = SESSION_STORE.get_session(session_id)
        if data and isinstance(data, dict):
            user_email = data.get("user_email")
            if user_email:
                logger.debug(f"Session found in Redis: {session_id} -> {user_email}")
                return user_email
    except Exception as e:
        logger.debug(f"Redis get_session failed: {e}")
    
    return None

def get_user_sessions(user_email: str) -> List[str]:
    """Return list of browser session IDs associated with a user."""

    # Prefer Redis because it is O(1) and globally consistent
    try:
        sessions = SESSION_STORE.get_user_sessions(user_email)
        if sessions:
            return sessions
    except Exception as e:
        logger.debug("Redis get_user_sessions failed: %s", e)

    # Fallback to sqlite
    conn = sqlite3.connect("user_data.db")
    c = conn.cursor()
    c.execute("SELECT session_id FROM browser_sessions WHERE user_email = ?", (user_email,))
    results = c.fetchall()
    conn.close()
    return [r[0] for r in results]

def delete_browser_session(session_id: str, user_email: str = None):
    """Delete session mapping from all stores."""

    # Redis first
    try:
        SESSION_STORE.delete_session(session_id)
        if user_email:
            SESSION_STORE.remove_user_session(user_email, session_id)
    except Exception as e:
        logger.debug("Redis delete_session failed: %s", e)

    # sqlite mirror
    conn = sqlite3.connect("user_data.db")
    c = conn.cursor()
    if user_email:
        c.execute(
            "DELETE FROM browser_sessions WHERE session_id = ? AND user_email = ?",
            (session_id, user_email),
        )
    else:
        c.execute("DELETE FROM browser_sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

def cleanup_old_sessions():
    """Clean up sessions older than 30 days"""
    conn = sqlite3.connect("user_data.db")
    c = conn.cursor()
    c.execute('''
        DELETE FROM browser_sessions 
        WHERE last_accessed < datetime('now', '-30 days')
    ''')
    conn.commit()
    conn.close()

# Ensure the sessions table exists as soon as the module is imported
init_browser_sessions_db()

def create_workflow():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("email_collection", email_collection_node)
    workflow.add_node("intent_detection", intent_detection_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("availability", availability_node)
    workflow.add_node("booking", booking_node)
    workflow.add_node("show_appointments", show_appointments_node)
    
    # ALWAYS start with email collection/authentication
    workflow.set_entry_point("email_collection")
    # From email collection, route based on authentication status
    workflow.add_conditional_edges(
        "email_collection",
        lambda state: (
            "end" if state.get('needs_more_info', False) or state.get('auth_required', False)
            else "intent_detection"  # Only proceed if authenticated and have email
        ),
        {
            "intent_detection": "intent_detection",
            "end": END
        }
    )
    
    workflow.add_edge("intent_detection", "validation")
    
    # Updated conditional routing from validation
    workflow.add_conditional_edges(
        "validation",
        lambda state: (
            "end" if state.get('auth_required', False) or state.get('needs_more_info', True)
            else "show_appointments" if state.get('intent') == 'show_appointments'
            else "availability" if state.get('intent') in ['book_appointment', 'check_availability']
            else "end"
        ),
        {
            "availability": "availability",
            "show_appointments": "show_appointments",
            "end": END
        }
    )
    
    workflow.add_edge("availability", "booking")
    workflow.add_edge("booking", END)
    workflow.add_edge("show_appointments", END)
    
    return workflow.compile()


graph = create_workflow()

@app.post("/store_session_user")
async def store_session_user(payload: dict, request: Request):
    """Store mapping of browser session (from cookie or payload) to user."""
    try:
        provided_session_id = payload.get("browser_session_id")
        session_id = resolve_browser_session_id(request, provided_session_id)
        user_email = payload.get("user_email")

        if session_id and user_email:
            store_browser_session(session_id, user_email)
            return {"status": "success"}
        else:
            return {"status": "error", "message": "Missing session_id or user_email"}
    except Exception as e:
        logger.exception("Error storing session user")
        return {"status": "error", "message": str(e)}

@app.get("/get_session_user")
async def get_session_user_endpoint(request: Request, browser_session_id: Optional[str] = None):
    """Get user email associated with the current browser session (fallback to query param)."""
    try:
        session_id = resolve_browser_session_id(request, browser_session_id)
        user_email = get_session_user(session_id)
        if user_email:
            # Check if user is still authenticated
            creds_json = load_user_credentials(user_email)
            authorized = False
            if creds_json:
                try:
                    service = get_calendar_service(user_email)
                    authorized = True
                except Exception:
                    delete_user_credentials(user_email)
            
            return {
                "user_email": user_email,
                "authorized": authorized
            }
        else:
            return {"user_email": None, "authorized": False}
    except Exception as e:
        logger.exception("Error getting session user")
        return {"user_email": None, "authorized": False, "error": str(e)}



@app.post("/initiate_auth")
async def initiate_auth(payload: dict, request: Request):
    """Enhanced auth initiation with proper session tracking"""
    try:
        user_email = payload.get("user_email")
        provided_session_id = payload.get("browser_session_id")
        
        # Get session ID from cookie or use provided one
        session_id = resolve_browser_session_id(request, provided_session_id)
        if not session_id:
            session_id = uuid.uuid4().hex
            logger.info(f"Generated new session ID for auth: {session_id}")
        
        if not user_email:
            return {"status": "error", "message": "user_email is required"}
        
        # Generate auth URL with session tracking
        auth_url = generate_auth_url(user_email, session_id)  # Pass session_id here
        
        # Store pending session in both systems
        browser_sessions[session_id] = {
            "user_email": user_email,
            "status": "pending_auth",
            "auth_url": auth_url
        }
        
        # Also store in Redis for persistence
        try:
            SESSION_STORE.set_session(session_id, {
                "user_email": user_email,
                "status": "pending_auth",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.warning(f"Redis storage failed during auth initiation: {e}")
        
        logger.info(f"Auth initiated for {user_email} with session: {session_id}")
        
        return {
            "status": "pending_auth",
            "auth_url": auth_url,
            "session_id": session_id,
            "message": f"Auth initiated for {user_email}"
        }
        
    except Exception as e:
        logger.exception("Error initiating auth")
        return {
            "status": "error",
            "auth_url": "",
            "message": f"Failed to initiate auth: {str(e)}"
        }

@app.get("/authenticated_users")
async def get_authenticated_users(request: Request, browser_session_id: Optional[str] = None):
    """Get list of users who have valid authentication for this SPECIFIC browser session"""
    try:
        browser_authenticated_users = []
        
        session_id = resolve_browser_session_id(request, browser_session_id)
        if session_id:
            user_email = get_session_user(session_id)
            if user_email:
                # Verify credentials are still valid
                creds_json = load_user_credentials(user_email)
                if creds_json:
                    try:
                        service = get_calendar_service(user_email)
                        browser_authenticated_users.append(user_email)
                    except Exception:
                        delete_user_credentials(user_email)
                        delete_browser_session(session_id, user_email)
        
        return {
            "browser_authenticated_users": browser_authenticated_users,
            "count": len(browser_authenticated_users)
        }
    except Exception as e:
        logger.exception("Error getting authenticated users")
        return {"error": str(e), "browser_authenticated_users": []}



@app.get("/user_status/{user_email}")
async def get_user_status(user_email: str):
    """Get detailed status for a specific user"""
    try:
        # Check if user has valid credentials
        creds_json = load_user_credentials(user_email)
        if not creds_json:
            return {
                "user_email": user_email,
                "authenticated": False,
                "status": "no_credentials"
            }
        
        # Try to get calendar service
        try:
            service = get_calendar_service(user_email)
            return {
                "user_email": user_email,
                "authenticated": True,
                "status": "valid_credentials",
                "can_access_calendar": True
            }
        except Exception as e:
            # Credentials exist but are invalid
            delete_user_credentials(user_email)
            return {
                "user_email": user_email,
                "authenticated": False,
                "status": "invalid_credentials",
                "error": str(e)
            }
    
    except Exception as e:
        logger.exception(f"Error checking status for {user_email}")
        return {
            "user_email": user_email,
            "authenticated": False,
            "status": "error",
            "error": str(e)
        }

@app.get("/database_status")
async def database_status():
    try:
        healthy = check_database_health()
        conn = sqlite3.connect("user_data.db")
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in c.fetchall()]
        counts = {tbl: c.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0] for tbl in tables}
        conn.close()
        return {
            "healthy": healthy,
            "tables": tables,
            "record_counts": counts,
            "in_memory_users": list(user_credentials.keys()),
        }
    except Exception as e:
        return {"error": str(e), "healthy": False}


@app.post("/fix_database")
async def fix_database():
    try:
        reset_database()
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/process_input")
async def process_input(req: BookingRequest):
    try:
        state = load_conversation_state(req.conversation_id)
        user_email = req.user_email or state.get("user_email")

        # Preserve the entire conversation context
        state.update(
            {
                "user_input": req.user_input,
                "conversation_id": req.conversation_id,
                "user_email": user_email,
                "intent": state.get("intent", ""),
                "date": state.get("date"),
                "time_range": state.get("time_range"),
                "specific_time": state.get("specific_time"),
                "available_slots": state.get("available_slots", []),
                "confirmed_slot": state.get("confirmed_slot", ""),
                "messages": state.get("messages", [])
                + [{"role": "user", "content": req.user_input}],
                "last_intent": state.get("last_intent", ""),
                "needs_more_info": state.get("needs_more_info", False),
                "auth_required": state.get("auth_required", False),
                "auth_url": state.get("auth_url", ""),
            }
        )

        final_state = graph.invoke(state)
        save_conversation_state(req.conversation_id, final_state)

        assistant_msgs = [m for m in final_state.get("messages", []) if m["role"] == "assistant"]
        reply = assistant_msgs[-1]["content"] if assistant_msgs else "I'm here to help!"
        return {"response": reply, "state": final_state}
    except Exception as e:
        logger.exception("process_input failed")
        return {"response": f"Error: {e}", "state": {}}


@app.post("/complete_auth")
async def complete_auth(user_email: str, auth_code: str):
    return {
        "status": "success" if complete_oauth_flow(user_email, auth_code) else "error"
    }

@app.get("/callback")
async def oauth_callback(request: Request, code: str, state: str):
    """Fixed OAuth callback with production cookie handling"""
    logger.info(f"OAuth callback - code: {code[:10]}..., state: {state}")
    
    session_id = state
    
    if not session_id:
        logger.error("No state parameter in OAuth callback")
        return RedirectResponse(f"{FRONTEND_URL}?error=no_state")
    
    # Find the user who initiated this OAuth flow
    user_email = None
    flow = None
    
    for email, data in user_credentials.items():
        if (data.get("status") == "pending_auth" and 
            data.get("session_id") == session_id):
            user_email = email
            flow = data.get("flow")
            break
    
    if not user_email or not flow:
        logger.error(f"No pending user found for session: {session_id}")
        return RedirectResponse(f"{FRONTEND_URL}?error=no_pending_auth")
    
    try:
        # Complete OAuth flow
        flow.fetch_token(code=code)
        store_user_credentials(user_email, flow.credentials)
        user_credentials[user_email]["status"] = "completed"
        
        # Store session mapping
        store_browser_session(session_id, user_email)
        
        logger.info(f"OAuth completed successfully: {session_id} -> {user_email}")
        
        # Create response with proper production cookie
        response = RedirectResponse(f"{FRONTEND_URL}?auth=success")
        
        # CRITICAL: Set cookie with production settings
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            max_age=SESSION_COOKIE_MAX_AGE,
            httponly=False,  # Frontend needs to read this
            samesite="lax",
            secure=IS_PRODUCTION,  # True in production
            path="/",
            domain=None  # Let browser handle domain
        )
        
        return response
        
    except Exception as e:
        logger.exception(f"OAuth callback failed for {user_email}: {e}")
        return RedirectResponse(f"{FRONTEND_URL}?error=oauth_failed")


@app.get("/debug_session/{session_id}")
async def debug_session(session_id: str):
    """Debug endpoint to check session state"""
    try:
        # Check Redis
        redis_data = {}
        try:
            redis_data = SESSION_STORE.get_session(session_id)
        except Exception as e:
            redis_data = {"error": str(e)}
        
        # Check SQLite
        sqlite_user = get_session_user(session_id)
        
        # Check in-memory
        memory_session = browser_sessions.get(session_id, {})
        
        return {
            "session_id": session_id,
            "redis_data": redis_data,
            "sqlite_user": sqlite_user,
            "memory_session": memory_session,
            "credentials_cache": list(user_credentials.keys())
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/validate_session")
async def validate_session(request: Request):
    """Redis-only session validation"""
    try:
        session_id = request.cookies.get(SESSION_COOKIE_NAME)
        logger.info(f"Validating session: {session_id}")
        
        if not session_id:
            return {"valid": False, "user_email": None, "reason": "no_cookie"}
        
        # Check Redis only
        user_email = get_session_user(session_id)
        
        if not user_email:
            logger.warning(f"No user mapping found for session: {session_id}")
            return {"valid": False, "user_email": None, "reason": "no_mapping"}
        
        # Verify credentials are still valid
        creds_json = load_user_credentials(user_email)
        if not creds_json:
            logger.warning(f"No credentials found for user: {user_email}")
            # Clean up Redis session
            SESSION_STORE.delete_session(session_id)
            SESSION_STORE.remove_user_session(user_email, session_id)
            return {"valid": False, "user_email": None, "reason": "no_credentials"}
        
        try:
            # Test calendar access
            service = get_calendar_service(user_email)
            
            # Refresh session TTL in Redis
            SESSION_STORE.set_session(session_id, {"user_email": user_email}, ttl=60*60*24*30)
            
            logger.info(f"Session validation successful for: {user_email}")
            return {
                "valid": True,
                "user_email": user_email,
                "session_id": session_id,
                "reason": "success"
            }
            
        except Exception as e:
            logger.error(f"Calendar service failed for {user_email}: {e}")
            # Invalid credentials - clean up
            delete_user_credentials(user_email)
            SESSION_STORE.delete_session(session_id)
            SESSION_STORE.remove_user_session(user_email, session_id)
            return {"valid": False, "user_email": None, "reason": "invalid_credentials"}
            
    except Exception as e:
        logger.exception("Session validation error")
        return {"valid": False, "user_email": None, "error": str(e), "reason": "exception"}


@app.get("/check_auth")
async def check_auth(request: Request, user_email: str, browser_session_id: Optional[str] = ""):
    """Enhanced check_auth that preserves session ID throughout OAuth flow"""
    try:
        # Resolve browser session ID from cookie or param
        session_id = resolve_browser_session_id(request, browser_session_id)
        
        # If no session ID exists, generate one and store it immediately
        if not session_id:
            session_id = uuid.uuid4().hex
            logger.info(f"Generated new session ID for auth check: {session_id}")
        
        # Check existing session mapping
        session_user = get_session_user(session_id) if session_id else None
        
        # If this session is already linked to another user, force fresh auth
        if session_id and session_user and session_user != user_email:
            auth_url = generate_auth_url(user_email)
            browser_sessions[session_id] = {
                "user_email": user_email,
                "status": "pending_auth",
                "auth_url": auth_url,
            }
            return {"authorized": False, "status": "pending_auth", "auth_url": auth_url}
        
        # Check existing credentials
        creds = load_user_credentials(user_email)
        if creds:
            try:
                service = get_calendar_service(user_email)
                session_meta = browser_sessions.get(session_id) if session_id else None
                
                if session_meta and session_meta.get("status") == "pending_auth":
                    # Still waiting for explicit OAuth completion in this browser.
                    return {
                        "authorized": False,
                        "status": "pending_auth",
                        "auth_url": session_meta.get("auth_url", ""),
                    }
                
                # Mark this session completed and persist mapping
                if session_id:
                    store_browser_session(session_id, user_email)
                    browser_sessions[session_id] = {
                        "user_email": user_email,
                        "status": "completed",
                        "auth_url": "",
                    }
                
                return {"authorized": True, "status": "completed", "auth_url": ""}
                
            except Exception:
                # Credentials invalid – force fresh OAuth
                delete_user_credentials(user_email)
                if session_id:
                    delete_browser_session(session_id, user_email)
        
        # Generate new auth URL and PRESERVE the session ID
        auth_url = generate_auth_url(user_email)
        if session_id:
            browser_sessions[session_id] = {
                "user_email": user_email,
                "status": "pending_auth",
                "auth_url": auth_url
            }
        
        return {
            "authorized": False,
            "status": "pending_auth",
            "auth_url": auth_url,
            "session_id": session_id  # Return session ID to frontend
        }
        
    except Exception as e:
        logger.exception(f"Error in check_auth for {user_email}")
        return {
            "authorized": False,
            "status": "error",
            "auth_url": "",
            "error": str(e)
        }


@app.post("/logout")
async def logout(payload: dict, request: Request):
    """Logout user from specific browser session"""
    try:
        user_email = payload.get("user_email")
        provided_session_id = payload.get("browser_session_id")
        browser_session_id = resolve_browser_session_id(request, provided_session_id)
        
        if browser_session_id:
            # Remove only this browser session
            delete_browser_session(browser_session_id, user_email)
            browser_sessions.pop(browser_session_id, None)
        else:
            # Fallback to original behavior - remove all user credentials
            delete_user_credentials(user_email)
            user_credentials.pop(user_email, None)
        
        return {"status": "success"}
    except Exception as e:
        logger.exception("Error in logout")
        return {"status": "error", "message": str(e)}


def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")

def enhanced_cleanup():
    """Enhanced cleanup for all session stores"""
    try:
        # Clean expired credentials
        cleanup_expired_credentials()
        
        # Clean old browser sessions
        cleanup_old_sessions()
        
        # Clean Redis sessions (if using Redis)
        if hasattr(SESSION_STORE, 'client'):
            # Clean expired Redis keys
            for key in SESSION_STORE.client.scan_iter(match="session:*"):
                ttl = SESSION_STORE.client.ttl(key)
                if ttl == -1:  # No expiry set
                    SESSION_STORE.client.expire(key, 60 * 60 * 24 * 7)
        
        logger.info("Session cleanup completed successfully")
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")

# Update the cleanup thread
def _periodic_cleanup():
    while True:
        time.sleep(300)  # 5 minutes
        enhanced_cleanup()  # Use the new enhanced cleanup

threading.Thread(target=_periodic_cleanup, daemon=True).start()

# ---------------------------------------------------------------------------
# Root path – redirect to interactive docs so hitting the base URL is useful
# ---------------------------------------------------------------------------


@app.get("/")
async def root():  # pragma: no cover – simple convenience route
    """Redirect bare URL to /docs swagger UI."""
    return RedirectResponse("/docs")

# ---------------------------------------------------------------------------
# Enhanced browser-session models (in-memory, optional to persist to DB)
# ---------------------------------------------------------------------------


class BrowserInfo(BaseModel):
    """Minimal fingerprint data sent by the front-end when it tries to restore a session."""

    user_agent: str = "unknown"
    screen_resolution: str = "unknown"
    timezone: str = "unknown"


class SessionData(BaseModel):
    """Lightweight session record kept in memory for quick lookup."""

    user_email: str | None = None
    browser_session_id: str
    authorized: bool = False
    created_at: datetime = datetime.utcnow()
    last_activity: datetime = datetime.utcnow()
    browser_fingerprint: str | None = None


def _generate_browser_fingerprint(info: BrowserInfo) -> str:
    """Create a stable hash from BrowserInfo fields."""

    raw = f"{info.user_agent}|{info.screen_resolution}|{info.timezone}"
    return hashlib.md5(raw.encode()).hexdigest()


# in-memory mappings (kept separate from the SQLite mapping we already use)
browser_session_meta: dict[str, SessionData] = {}


def _cleanup_expired_browser_meta() -> None:
    """Remove sessions older than 30 days from the in-memory map."""

    cutoff = datetime.utcnow() - timedelta(days=30)
    for sid in list(browser_session_meta.keys()):
        if browser_session_meta[sid].last_activity < cutoff:
            browser_session_meta.pop(sid, None)


# ---------------------------------------------------------------------------
# Admin / helper endpoints for session debugging
# ---------------------------------------------------------------------------


@app.get("/get_browser_sessions")
async def get_browser_sessions():
    """Return in-memory browser sessions (debug)."""

    _cleanup_expired_browser_meta()
    payload: list[dict[str, Any]] = []
    for sid, meta in browser_session_meta.items():
        payload.append(
            {
                "session_id": sid,
                "user_email": meta.user_email,
                "authorized": meta.authorized,
                "created_at": meta.created_at.isoformat(),
                "last_activity": meta.last_activity.isoformat(),
            }
        )
    return {"sessions": payload}


@app.post("/restore_session")
async def restore_session(info: BrowserInfo):
    """Attempt to restore session by browser fingerprint after hard refresh."""

    _cleanup_expired_browser_meta()
    fp = _generate_browser_fingerprint(info)
    for sid, meta in browser_session_meta.items():
        if meta.browser_fingerprint == fp and meta.authorized and meta.user_email:
            meta.last_activity = datetime.utcnow()
            browser_session_meta[sid] = meta
            return {
                "session_found": True,
                "session_id": sid,
                "user_email": meta.user_email,
                "authorized": True,
            }

    return {"session_found": False}


# ---------------------------------------------------------------------------
# Redis connection health-check endpoint
# ---------------------------------------------------------------------------

@app.get("/redis_test")
async def redis_test():
    """Provides a direct way to test Redis connectivity and configuration."""
    
    # Check if we are using the real store or the dummy one
    from agent.utils.session_store import RedisSessionStore, _DummyStore
    
    if isinstance(SESSION_STORE, _DummyStore):
        return {
            "status": "CONFIG_ERROR",
            "message": "Redis is not configured. The application is using a no-op dummy store. Check startup logs for 'FAILED to initialize RedisSessionStore'.",
            "troubleshooting": {
                "check_1": "Is REDIS_URL set in the environment?",
                "check_2": "Is SESSION_ENCRYPTION_KEY set and a valid 44-character Fernet key?",
                "check_3": "Are 'redis' and 'cryptography' packages installed?",
                "check_4": "Can the backend container connect to the Redis host/port specified in REDIS_URL?",
            }
        }
    
    if isinstance(SESSION_STORE, RedisSessionStore):
        try:
            test_key = "redis_test_key"
            test_value = f"Connection successful at {datetime.utcnow().isoformat()}"
            SESSION_STORE.client.set(test_key, test_value, ex=60) # set with 1-minute expiry
            read_value = SESSION_STORE.client.get(test_key)
            
            if read_value == test_value:
                 return {
                    "status": "SUCCESS",
                    "message": "Successfully connected to Redis, wrote a key, and read it back.",
                    "redis_ping": SESSION_STORE.client.ping(),
                 }
            else:
                 return {
                    "status": "ERROR",
                    "message": "Connected to Redis, but failed to read back the key that was just written.",
                    "wrote": test_value,
                    "read": read_value
                 }

        except Exception as e:
            return {
                "status": "CONNECTION_ERROR",
                "message": "Failed to communicate with Redis server even though the store was initialized.",
                "error_type": str(type(e).__name__),
                "error_details": str(e),
            }

    return {"status": "UNKNOWN_ERROR", "message": "The session store is of an unexpected type."}

