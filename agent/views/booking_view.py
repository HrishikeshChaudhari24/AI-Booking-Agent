import time
import uuid
import requests
import streamlit as st
import urllib.parse
import hashlib
import json
from datetime import datetime, timedelta
import streamlit.components.v1 as components
import os

# ---------------------------------------------------------------------------
# Ensure parent project root is on PYTHONPATH so `import agent...` works when
# Streamlit launches this file from inside agent/views.
# ---------------------------------------------------------------------------

import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

BACKEND = st.secrets["default"]["BACKEND_BASE_URL"] 
# Cookie keys for persisting session data
USER_EMAIL_COOKIE = "booking_user_email"
SESSION_ID_COOKIE = "booking_session_id"

# ---------------------------------------------------------------------------
# Cookie Management (using streamlit-cookies-manager)
# ---------------------------------------------------------------------------
# Placeholder for streamlit-cookies-manager, assuming it's added to requirements
# In a real scenario, you would import it:
# from streamlit_cookies_manager import EncryptedCookieManager
# For now, we will use st.session_state as a mock
# ---------------------------------------------------------------------------

# This is a simplified placeholder. In a real app, you'd use a robust library.
# import streamlit as st
# import streamlit.components.v1 as components
# import os

class SimpleCookieManager:
    def __init__(self, key):
        self._key = key
        self._session_cookie_name = "browser_session_id"  # Define directly here
        
        if 'cookies' not in st.session_state:
            st.session_state['cookies'] = {}
        
        # Sync with browser cookies on startup
        self._sync_browser_cookies()
    
    def _sync_browser_cookies(self):
        """Sync session cookie from browser to Streamlit session state"""
        
        # JavaScript to read browser cookie and pass to Streamlit
        cookie_sync_js = f"""
        <script>
        function getCookie(name) {{
            let cookieValue = null;
            if (document.cookie && document.cookie !== '') {{
                const cookies = document.cookie.split(';');
                for (let i = 0; i < cookies.length; i++) {{
                    const cookie = cookies[i].trim();
                    if (cookie.substring(0, name.length + 1) === (name + '=')) {{
                        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                        break;
                    }}
                }}
            }}
            return cookieValue;
        }}
        
        const sessionCookie = getCookie('{self._session_cookie_name}');
        if (sessionCookie) {{
            // Store in Streamlit session state
            window.parent.postMessage({{
                type: 'session_cookie',
                value: sessionCookie
            }}, '*');
        }}
        </script>
        """
        
        # Execute JavaScript to read cookie (silently)
        try:
            components.html(cookie_sync_js, height=0)
        except Exception as e:
            # Ignore JavaScript execution errors in development
            pass
    
    def get(self, cookie, default=None):
        return st.session_state.cookies.get(cookie, default)
    
    def set(self, cookie, value, expires_at=None, key=None):
        st.session_state.cookies[cookie] = value
        
        # Also set in browser via JavaScript
        is_production = os.getenv('ENV', '').lower() == 'production'
        secure_flag = 'Secure;' if is_production else ''
        
        set_cookie_js = f"""
        <script>
        document.cookie = "{cookie}={value}; path=/; max-age=2592000; SameSite=lax; {secure_flag}";
        </script>
        """
        
        try:
            components.html(set_cookie_js, height=0)
        except Exception as e:
            # Ignore JavaScript execution errors
            pass

    def delete(self, cookie, key=None):
        if cookie in st.session_state.cookies:
            del st.session_state.cookies[cookie]
        
        # Also delete from browser
        delete_cookie_js = f"""
        <script>
        document.cookie = "{cookie}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT;";
        </script>
        """
        
        try:
            components.html(delete_cookie_js, height=0)
        except Exception as e:
            pass

# This should be initialized once at the top
# The key should be a secret for encryption
cookies = SimpleCookieManager(key="dummy_key_for_now")

# ADD THIS: JavaScript component to read real browser cookies
# ADD THIS: JavaScript component to read real browser cookies
def sync_browser_cookie(cookie_name: str):
    """
    Uses a JavaScript component to read a real browser cookie and sync it
    into st.session_state for the mock SimpleCookieManager to use.
    """
    # Only run this synchronization logic once per session
    if f"{cookie_name}_synced" not in st.session_state:
        # This JS code runs in the user's browser. It reads the cookie
        # and reloads the page once with the cookie value in a query parameter.
        js_code = f"""
        <script>
            function getCookie(name) {{
                let cookieValue = null;
                if (document.cookie && document.cookie !== '') {{
                    const cookies = document.cookie.split(';');
                    for (let i = 0; i < cookies.length; i++) {{
                        const cookie = cookies[i].trim();
                        if (cookie.substring(0, name.length + 1) === (name + '=')) {{
                            cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                            break;
                        }}
                    }}
                }}
                return cookieValue;
            }}

            const cookieVal = getCookie('{cookie_name}');
            // If the cookie exists and we haven't already tried to sync it via URL
            if (cookieVal) {{
                const params = new URLSearchParams(window.location.search);
                if (!params.has('session_synced')) {{
                    params.set('session_synced', cookieVal);
                    // Reload the page to send the value to Streamlit
                    window.location.search = params.toString();
                }}
            }}
        </script>
        """
        components.html(js_code, height=0)

        # After the JS runs and reloads, this Python code will execute.
        # It checks for the query parameter set by the JavaScript.
        synced_val = st.query_params.get("session_synced")
        if synced_val:
            # Inject the real cookie value into the mock manager's state
            st.session_state['cookies'][cookie_name] = synced_val
            
            # Mark as synced to prevent this logic from running again
            st.session_state[f"{cookie_name}_synced"] = True
            
            # Clean up the URL for a better user experience
            st.query_params.clear()

def generate_browser_session_id():
    """Generate a unique session ID for this browser"""
    import random
    import string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=32))

# ---------------------------------------------------------------------------
# Enhanced Authentication Functions
# ---------------------------------------------------------------------------

def get_authenticated_users_for_browser():
    """Get list of users authenticated in this specific browser session ONLY"""
    if 'browser_session_id' not in st.session_state:
        return []
    
    try:
        response = requests.get(
            f"{BACKEND}/authenticated_users",
            params={"browser_session_id": st.session_state.browser_session_id},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("browser_authenticated_users", [])
    except Exception as e:
        st.error(f"Error checking authenticated users: {e}")
    return []

def validate_user_with_backend(user_email: str) -> dict:
    """Validate user authentication with backend and get auth status"""
    try:
        params = {
            "user_email": user_email,
            "browser_session_id": st.session_state.get('browser_session_id', '')
        }
        response = requests.get(
            f"{BACKEND}/check_auth",
            params=params,
            timeout=5
        )
        if response.status_code == 200:
            result = response.json()
            return {
                'authorized': result.get('authorized', False),
                'status': result.get('status', 'unknown'),
                'auth_url': result.get('auth_url', '')
            }
    except Exception as e:
        st.error(f"Error validating user: {e}")
    return {'authorized': False, 'status': 'error', 'auth_url': ''}

def initiate_auth_for_user(user_email: str) -> dict:
    """Initiate authentication flow for a user"""
    try:
        data = {
            "user_email": user_email,
            "browser_session_id": st.session_state.get('browser_session_id', '')
        }
        response = requests.post(
            f"{BACKEND}/initiate_auth",
            json=data,
            timeout=5
        )
        if response.status_code == 200:
            result = response.json()
            return {
                'success': True,
                'auth_url': result.get('auth_url', ''),
                'status': result.get('status', 'pending_auth')
            }
    except Exception as e:
        st.error(f"Error initiating auth: {e}")
    return {'success': False, 'auth_url': '', 'status': 'error'}

# ---------------------------------------------------------------------------
# Session Initialization and Restoration
# ---------------------------------------------------------------------------

def restore_or_initialize_session():
    """
    Enhanced session restoration with backend validation
    """
    sync_browser_cookie(cookie_name=SESSION_ID_COOKIE)# ADD THIS LINE TO SYNC REAL COOKIE
    if "session_initialized" in st.session_state:
        return

    # 1. Get or create the browser session ID
    session_id = cookies.get(SESSION_ID_COOKIE)
    if not session_id:
        session_id = generate_browser_session_id()
        cookies.set(SESSION_ID_COOKIE, session_id)
    st.session_state.browser_session_id = session_id

    # 2. Try to validate existing session with backend
    try:
        # First check the simple endpoint
        res = requests.get(
            f"{BACKEND}/get_session_user",
            params={"browser_session_id": session_id},
            timeout=5,
        )
        
        if res.status_code == 200:
            data = res.json()
            user_email = data.get("user_email")
            authorized = data.get("authorized", False)
            
            if user_email and authorized:
                # Double-check with validation endpoint
                validation_res = requests.get(
                    f"{BACKEND}/validate_session",
                    cookies={"booking_session": session_id},
                    timeout=5
                )
                
                if validation_res.status_code == 200:
                    validation_data = validation_res.json()
                    if validation_data.get("valid"):
                        st.session_state.user_email = user_email
                        st.session_state.authenticated = True
                        st.session_state.auth_required = False
                        cookies.set(USER_EMAIL_COOKIE, user_email)
                        st.success(f"✅ Session restored for {user_email}")
                        st.session_state.session_initialized = True
                        return
                    else:
                        st.warning(f"Session validation failed: {validation_data.get('reason', 'unknown')}")
                        
            # Clear invalid session data
            st.session_state.user_email = None
            st.session_state.authenticated = False
            cookies.delete(USER_EMAIL_COOKIE)
            
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not connect to backend: {e}")
        st.session_state.authenticated = False

    # Mark session as initialized
    st.session_state.session_initialized = True

def handle_streamlit_auth() -> bool:
    """Handles the full auth flow within Streamlit, now simplified."""
    if st.session_state.get("auth_required") and st.session_state.get("auth_url"):
        st.warning("🔐 Google Calendar authorization required")
        
        # Add session info to auth URL for better callback handling
        auth_url = st.session_state.auth_url
        if "?" in auth_url:
            auth_url += f"&session_id={st.session_state.browser_session_id}&user_email={urllib.parse.quote(st.session_state.user_email)}"
        else:
            auth_url += f"?session_id={st.session_state.browser_session_id}&user_email={urllib.parse.quote(st.session_state.user_email)}"
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            **Step 1:** [Click here to authorize Google Calendar access]({auth_url})
            
            **Step 2:** Complete the authorization in the new tab
            
            **Step 3:** Return to this page and click "Check Authorization"
            """)
        
        with col2:
            if st.button("🔄 Check Authorization", key="manual_check"):
                try:
                    auth_result = validate_user_with_backend(st.session_state.user_email)
                    if auth_result['authorized']:
                        st.success("✅ Authorization completed!")
                        st.session_state.auth_required = False
                        st.session_state.auth_url = ""
                        # Ensure session is saved
                        cookies.set(USER_EMAIL_COOKIE, st.session_state.user_email)
                        cookies.set(SESSION_ID_COOKIE, st.session_state.browser_session_id)
                        st.rerun()
                    else:
                        st.info(f"Status: {auth_result.get('status', 'pending')}")
                        st.info("Please complete the authorization in the other tab first.")
                except Exception as e:
                    st.error(f"Error checking auth status: {e}")
        
        return True
    return False
    


def render_login_interface():
    """Render the login interface in sidebar - SECURE VERSION"""
    st.subheader("🔐 Account Login")
    
    # Get authenticated users for THIS specific browser only
    authenticated_users = get_authenticated_users_for_browser()
    
    if authenticated_users:
        # Show quick login for users authenticated in THIS browser
        st.markdown("**Your authenticated accounts:**")
        for i, user in enumerate(authenticated_users):
            if st.button(f"📧 {user}", key=f"quick_login_{i}", use_container_width=True):
                st.session_state.user_email = user
                # Verify authentication for this specific browser session
                auth_result = validate_user_with_backend(user)
                if auth_result['authorized']:
                    st.session_state.auth_required = False
                    # Store in browser storage
                    cookies.set(USER_EMAIL_COOKIE, user)
                    st.success(f"Logged in as {user}")
                    st.rerun()
                else:
                    st.session_state.auth_required = True
                    st.session_state.auth_url = auth_result.get('auth_url', '')
                    if not st.session_state.auth_url:
                        auth_init = initiate_auth_for_user(user)
                        st.session_state.auth_url = auth_init.get('auth_url', '')
                    st.warning("Re-authentication required")
                    st.rerun()
        
        st.markdown("---")
        
    # Always show option to add new account
    st.markdown("**Add new account:**")
    new_email = st.text_input(
        "Email address:",
        placeholder="your.email@gmail.com",
        key="new_email_input",
        label_visibility="collapsed"
    )
    
    if st.button("Continue with Email", disabled=not new_email, use_container_width=True):
        if new_email and "@" in new_email:
            st.session_state.user_email = new_email
            # Store email immediately
            cookies.set(USER_EMAIL_COOKIE, new_email)
            # Always require fresh authentication for new emails
            st.session_state.auth_required = True
            auth_init = initiate_auth_for_user(new_email)
            st.session_state.auth_url = auth_init.get('auth_url', '')
            st.info("Authentication required for this account")
            st.rerun()
        else:
            st.error("Please enter a valid email address")

# ---------------------------------------------------------------------------
# Main Streamlit Application
# ---------------------------------------------------------------------------

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="AI Booking Agent", page_icon="📅")

    # --- Session Initialization (using the new, working logic) ---
    restore_or_initialize_session()

    # Initialize session state variables if they don't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "auth_url" not in st.session_state:
        st.session_state.auth_url = ""

    st.title("📅 AI Appointment Assistant")

    # --- Sidebar (restored to the user's desired layout) ---
    with st.sidebar:
        if st.session_state.get("user_email"):
            # User is logged in - show status and logout
            st.success(f"**Logged in as:**\n{st.session_state.user_email}")
            st.info(f"**Browser Session:** {st.session_state.get('browser_session_id', 'N/A')[:8]}...")
            
            if st.session_state.get("auth_required"):
                st.warning("⚠️ Authorization required")
            else:
                st.success("✅ Ready to help!")
            
            # Account actions
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Refresh", key="refresh_auth", use_container_width=True):
                    auth_result = validate_user_with_backend(st.session_state.user_email)
                    if auth_result['authorized']:
                        st.session_state.auth_required = False
                        st.session_state.auth_url = ""
                        st.success("✅ Refreshed!")
                    else:
                        st.session_state.auth_required = True
                        st.session_state.auth_url = auth_result.get('auth_url', '')
                        st.warning("Re-auth needed")
                    st.rerun()
            
            with col2:
                if st.button("🔒 Logout", key="logout_btn", use_container_width=True):
                    try:
                        requests.post(
                            f"{BACKEND}/logout",
                            json={
                                "user_email": st.session_state.user_email,
                                "browser_session_id": st.session_state.browser_session_id
                            },
                            timeout=5,
                        )
                    except Exception as e:
                        st.error(f"Logout error: {e}")
                    
                    # Use the new cookie manager
                    cookies.delete(USER_EMAIL_COOKIE)
                    
                    # Clear session state
                    st.session_state.user_email = None
                    st.session_state.authenticated = False
                    st.session_state.auth_required = False
                    st.session_state.auth_url = ""
                    st.session_state.messages = []
                    st.session_state.conversation_id = str(uuid.uuid4())
                    st.rerun()
        else:
            # User not logged in - show login interface
            render_login_interface()

        # Debug info in sidebar (restored)
        st.markdown("---")
        if st.checkbox("🔧 Debug Info", key="debug_toggle"):
            st.subheader("Debug Information")
            st.json({
                "user_email": st.session_state.user_email,
                "authenticated": st.session_state.get("authenticated"),
                "auth_required": st.session_state.get("auth_required"),
                "has_auth_url": bool(st.session_state.get("auth_url")),
                "conversation_id": st.session_state.conversation_id,
                "browser_session_id": st.session_state.get('browser_session_id', 'None'),
                "message_count": len(st.session_state.messages),
            })


    # --- Main Panel Logic (restored to the user's desired layout) ---
    if handle_streamlit_auth():
        return

    if not st.session_state.user_email:
        # Welcome screen for new users
        st.markdown("""
        ### Welcome! 👋
        
        I'm your AI appointment assistant. I can help you:
        - 📅 **Book appointments**
        - 🔍 **Check availability**
        - 📋 **View your schedule**
        
        **To get started, please log in using the sidebar** 👈
        """)
        return

    if st.session_state.user_email and not st.session_state.auth_required:
        # Chat Interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_input := st.chat_input("Ask me to book an appointment or check availability..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    response = requests.post(
                        f"{BACKEND}/process_input",
                        json={
                            "user_input": user_input,
                            "conversation_id": st.session_state.conversation_id,
                            "user_email": st.session_state.user_email,
                        },
                        timeout=60,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        backend_state = result.get("state", {})
                        st.session_state.auth_required = backend_state.get("auth_required", False)
                        st.session_state.auth_url = backend_state.get("auth_url", "")

                        if not st.session_state.auth_required:
                            message_placeholder.markdown(result["response"])
                            st.session_state.messages.append({"role": "assistant", "content": result["response"]})
                        else:
                            st.rerun()
                    else:
                        message_placeholder.error(f"Error: {response.text}")
                except Exception as e:
                    message_placeholder.error(f"An error occurred: {e}")

# ---------------------------------------------------------------------------
# Ensure FastAPI backend is running when deployed on Streamlit Cloud
# ---------------------------------------------------------------------------

if os.getenv("START_BACKEND_WITH_STREAMLIT", "0") == "1" and "_backend_started" not in st.session_state:
    try:
        from threading import Thread
        from agent.controllers.booking_controller import run_fastapi

        Thread(target=run_fastapi, daemon=True).start()
        st.session_state._backend_started = True
        # Give the server a moment to bind
        time.sleep(1)
    except Exception as e:
        st.warning(f"Could not start backend automatically: {e}")

if __name__ == "__main__":
    main()
