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
# Improved Cookie Management Functions
# ---------------------------------------------------------------------------

def set_persistent_cookie(key: str, value: str, days: int = 365):
    """Set a persistent cookie that survives browser restarts"""
    cookie_script = f"""
    <script>
    function setPersistentCookie(name, value, days) {{
        var expires = "";
        if (days) {{
            var date = new Date();
            date.setTime(date.getTime() + (days*24*60*60*1000));
            expires = "; expires=" + date.toUTCString();
        }}
        document.cookie = name + "=" + encodeURIComponent(value || "") + expires + "; path=/; SameSite=Lax";
        console.log("Set cookie:", name, "=", value);
    }}
    setPersistentCookie("{key}", "{value}", {days});
    </script>
    """
    components.html(cookie_script, height=0)

def get_persistent_cookie_sync(key: str) -> str:
    """Synchronously get cookie value using localStorage as fallback"""
    # Use a unique key for this request
    request_id = str(uuid.uuid4())
    
    cookie_script = f"""
    <script>
    function getCookieValue(name) {{
        var nameEQ = name + "=";
        var ca = document.cookie.split(';');
        for(var i=0;i < ca.length;i++) {{
            var c = ca[i];
            while (c.charAt(0)==' ') c = c.substring(1,c.length);
            if (c.indexOf(nameEQ) == 0) return decodeURIComponent(c.substring(nameEQ.length,c.length));
        }}
        return null;
    }}
    
    // Try to get from cookie first
    var cookieValue = getCookieValue("{key}");
    
    // If no cookie, try localStorage as backup
    if (!cookieValue) {{
        try {{
            cookieValue = localStorage.getItem("{key}");
        }} catch (e) {{
            console.log("localStorage not available");
        }}
    }}
    
    // Store in sessionStorage for immediate retrieval
    if (cookieValue) {{
        try {{
            sessionStorage.setItem("temp_{request_id}", cookieValue);
        }} catch (e) {{
            console.log("sessionStorage not available");
        }}
    }}
    
    console.log("Retrieved {key}:", cookieValue);
    </script>
    """
    components.html(cookie_script, height=0)
    
    # Small delay to allow script execution
    time.sleep(0.1)
    
    # Try to retrieve from session storage
    retrieve_script = f"""
    <script>
    try {{
        var value = sessionStorage.getItem("temp_{request_id}");
        if (value) {{
            // Pass to Streamlit via a div element
            var div = document.createElement('div');
            div.id = 'cookie_result_{request_id}';
            div.style.display = 'none';
            div.textContent = value;
            document.body.appendChild(div);
            
            // Clean up
            sessionStorage.removeItem("temp_{request_id}");
        }}
    }} catch (e) {{
        console.log("Error retrieving from sessionStorage");
    }}
    </script>
    """
    components.html(retrieve_script, height=0)
    
    return ""

def store_in_browser_storage(key: str, value: str):
    """Store value in both cookies and localStorage for redundancy"""
    storage_script = f"""
    <script>
    // Store in cookie
    function setPersistentCookie(name, value, days) {{
        var expires = "";
        if (days) {{
            var date = new Date();
            date.setTime(date.getTime() + (days*24*60*60*1000));
            expires = "; expires=" + date.toUTCString();
        }}
        document.cookie = name + "=" + encodeURIComponent(value || "") + expires + "; path=/; SameSite=Lax";
    }}
    
    // Store in localStorage as backup
    try {{
        localStorage.setItem("{key}", "{value}");
    }} catch (e) {{
        console.log("localStorage not available");
    }}
    
    setPersistentCookie("{key}", "{value}", 365);
    console.log("Stored {key} in both cookie and localStorage");
    </script>
    """
    components.html(storage_script, height=0)

def clear_browser_storage(key: str):
    """Clear value from both cookies and localStorage"""
    clear_script = f"""
    <script>
    // Clear cookie
    document.cookie = "{key}=; expires=Thu, 01 Jan 1970 00:00:01 GMT; path=/";
    
    // Clear localStorage
    try {{
        localStorage.removeItem("{key}");
    }} catch (e) {{
        console.log("localStorage not available");
    }}
    
    console.log("Cleared {key} from browser storage");
    </script>
    """
    components.html(clear_script, height=0)

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

def initialize_browser_session():
    """Initialize browser session with enhanced persistence"""
    # Check if we already have session data
    if 'session_initialized' in st.session_state:
        return
    
    # Try to restore from URL parameters first (for auth callbacks)
    try:
        if hasattr(st, "query_params"):
            qp = st.query_params
            if "session_id" in qp and "user_email" in qp:
                st.session_state.browser_session_id = qp["session_id"]
                st.session_state.user_email = qp["user_email"]
                # Clear URL parameters
                qp.clear()
                # Store in browser for next time
                store_in_browser_storage(SESSION_ID_COOKIE, st.session_state.browser_session_id)
                store_in_browser_storage(USER_EMAIL_COOKIE, st.session_state.user_email)
                st.session_state.session_initialized = True
                return
    except Exception:
        pass
    
    # Generate or retrieve browser session ID
    if 'browser_session_id' not in st.session_state:
        # For production, try to get from query params or backend first
        try:
            # Check if backend has any existing sessions for this browser
            response = requests.get(f"{BACKEND}/get_browser_sessions", timeout=5)
            if response.status_code == 200:
                sessions = response.json().get('sessions', [])
                if sessions:
                    # Use the most recent session
                    latest_session = max(sessions, key=lambda x: x.get('created_at', ''))
                    st.session_state.browser_session_id = latest_session['session_id']
                    if latest_session.get('user_email'):
                        st.session_state.user_email = latest_session['user_email']
        except Exception:
            pass
        
        # If still no session ID, generate new one
        if 'browser_session_id' not in st.session_state:
            st.session_state.browser_session_id = generate_browser_session_id()
        
        # Store in browser storage
        store_in_browser_storage(SESSION_ID_COOKIE, st.session_state.browser_session_id)
    
    # Try to restore user email if not already set
    if 'user_email' not in st.session_state or not st.session_state.user_email:
        # Check with backend if this session has a user
        try:
            response = requests.get(
                f"{BACKEND}/get_session_user",
                params={"browser_session_id": st.session_state.browser_session_id},
                timeout=5
            )
            if response.status_code == 200:
                result = response.json()
                if result.get('user_email'):
                    st.session_state.user_email = result['user_email']
                    st.session_state.auth_required = not result.get('authorized', False)
                    # Store email in browser storage for next time
                    store_in_browser_storage(USER_EMAIL_COOKIE, st.session_state.user_email)
        except Exception:
            pass
    
    # Validate current authentication status if we have a user
    if st.session_state.get('user_email'):
        try:
            auth_result = validate_user_with_backend(st.session_state.user_email)
            st.session_state.auth_required = not auth_result.get('authorized', False)
            st.session_state.auth_url = auth_result.get('auth_url', '')
        except Exception:
            st.session_state.auth_required = True
    
    st.session_state.session_initialized = True

def restore_session_from_backend():
    """Try to restore session state from backend using browser fingerprint"""
    try:
        # Create a browser fingerprint using available data
        browser_info = {
            "user_agent": st.query_params.get("user_agent", "unknown"),
            "screen_resolution": st.query_params.get("screen", "unknown"),
            "timezone": st.query_params.get("tz", "unknown")
        }
        
        response = requests.post(
            f"{BACKEND}/restore_session",
            json=browser_info,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('session_found'):
                st.session_state.browser_session_id = result['session_id']
                st.session_state.user_email = result.get('user_email')
                st.session_state.auth_required = not result.get('authorized', False)
                
                # Store restored data in browser
                store_in_browser_storage(SESSION_ID_COOKIE, st.session_state.browser_session_id)
                if st.session_state.user_email:
                    store_in_browser_storage(USER_EMAIL_COOKIE, st.session_state.user_email)
                
                return True
    except Exception:
        pass
    return False

def handle_streamlit_auth() -> bool:
    """Handle the authentication flow"""
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
                        store_in_browser_storage(USER_EMAIL_COOKIE, st.session_state.user_email)
                        store_in_browser_storage(SESSION_ID_COOKIE, st.session_state.browser_session_id)
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
                    store_in_browser_storage(USER_EMAIL_COOKIE, user)
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
            store_in_browser_storage(USER_EMAIL_COOKIE, new_email)
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
    st.set_page_config(page_title="AI Appointment Assistant", page_icon="📅")

    # Initialize browser session first
    initialize_browser_session()

    # Initialize session state
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    if "auth_required" not in st.session_state:
        st.session_state.auth_required = False
    if "auth_url" not in st.session_state:
        st.session_state.auth_url = ""

    # Handle URL parameters for auth success/error
    try:
        if hasattr(st, "query_params"):
            qp = st.query_params
            if "auth" in qp and qp["auth"] == "success":
                st.success("✅ Authorization successful! You can now book appointments.")
                st.session_state.auth_required = False
                st.session_state.auth_url = ""
                # Ensure session is saved
                if st.session_state.user_email:
                    store_in_browser_storage(USER_EMAIL_COOKIE, st.session_state.user_email)
                    store_in_browser_storage(SESSION_ID_COOKIE, st.session_state.browser_session_id)
                qp.clear()
                st.rerun()
            elif "error" in qp:
                st.error(f"❌ Authorization failed: {qp['error']}")
                st.session_state.auth_required = False
                st.session_state.auth_url = ""
                qp.clear()
    except Exception:
        pass

    st.title("📅 AI Appointment Assistant")

    # Sidebar for user authentication
    with st.sidebar:
        if st.session_state.user_email:
            # User is logged in - show status and logout
            st.success(f"**Logged in as:**\n{st.session_state.user_email}")
            st.info(f"**Browser Session:** {st.session_state.browser_session_id[:8]}...")
            
            if st.session_state.auth_required:
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
                        # Logout from this browser session
                        response = requests.post(
                            f"{BACKEND}/logout",
                            json={
                                "user_email": st.session_state.user_email,
                                "browser_session_id": st.session_state.browser_session_id
                            },
                            timeout=5,
                        )
                    except Exception as e:
                        st.error(f"Logout error: {e}")

                    # Clear browser storage
                    clear_browser_storage(USER_EMAIL_COOKIE)
                    clear_browser_storage(SESSION_ID_COOKIE)
                    
                    # Clear session state
                    st.session_state.user_email = None
                    st.session_state.auth_required = False
                    st.session_state.auth_url = ""
                    st.session_state.messages = []
                    st.session_state.conversation_id = str(uuid.uuid4())
                    # Generate new browser session ID
                    st.session_state.browser_session_id = generate_browser_session_id()
                    store_in_browser_storage(SESSION_ID_COOKIE, st.session_state.browser_session_id)
                    
                    st.success("👋 Logged out!")
                    st.rerun()
        else:
            # User not logged in - show login interface
            render_login_interface()

    # Handle authentication flow
    if handle_streamlit_auth():
        return

    # Show appropriate content based on login status
    if not st.session_state.user_email:
        # Welcome screen for new users
        st.markdown("""
        ### Welcome! 👋
        
        I'm your AI appointment assistant. I can help you:
        
        - 📅 **Book appointments** - Just say "Book a meeting tomorrow at 3 PM"
        - 🔍 **Check availability** - Ask "Am I free Friday afternoon?"
        - 📋 **View your schedule** - Say "Show my appointments"
        
        **To get started, please log in using the sidebar** 👈
        """)
        
        # Add some example interactions
        with st.expander("💡 See example conversations"):
            st.markdown("""
            **Booking an appointment:**
            - "Book a meeting tomorrow at 2 PM"
            - "Schedule a call with John next Friday from 10-11 AM"
            
            **Checking availability:**
            - "Am I free this Thursday afternoon?"
            - "What's my availability next week?"
            
            **Viewing schedule:**
            - "Show my appointments"
            - "What meetings do I have today?"
            """)
        
        return

    # Main chat interface - only show if user is selected
    if st.session_state.user_email and not st.session_state.auth_required:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if user_input := st.chat_input("Ask me to book an appointment or check availability..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Process user input
            try:
                with st.spinner("Thinking..."):
                    response = requests.post(
                        f"{BACKEND}/process_input",
                        json={
                            "user_input": user_input,
                            "conversation_id": st.session_state.conversation_id,
                            "user_email": st.session_state.user_email,
                        },
                        timeout=30,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        backend_state = result.get("state", {})
                        
                        # Update auth state from backend
                        st.session_state.auth_required = backend_state.get("auth_required", False)
                        st.session_state.auth_url = backend_state.get("auth_url", "")

                        if not st.session_state.auth_required:
                            # Show assistant response
                            with st.chat_message("assistant"):
                                st.markdown(result["response"])
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": result["response"]
                            })
                        else:
                            # Need to re-authenticate
                            st.rerun()
                    else:
                        error_msg = "Sorry, I'm having trouble processing your request. Please try again."
                        with st.chat_message("assistant"):
                            st.markdown(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })

            except requests.exceptions.RequestException:
                error_msg = "Connection error. Please make sure the server is running and try again."
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
            except Exception as e:
                error_msg = f"An unexpected error occurred: {str(e)}"
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

    # Show welcome message for authenticated users with empty chat
    elif st.session_state.user_email and not st.session_state.auth_required and not st.session_state.messages:
        st.success(f"✅ Ready to help! Authenticated as: {st.session_state.user_email}")
        st.markdown("""
        ### What would you like to do?
        
        You can ask me to:
        - Book a new appointment
        - Check your availability
        - Show your upcoming meetings
        
        Just type your request below! 👇
        """)

    # Debug info in sidebar (optional - remove in production)
    with st.sidebar:
        if st.checkbox("🔧 Debug Info", key="debug_toggle"):
            st.markdown("---")
            st.subheader("Debug Information")
            st.json({
                "user_email": st.session_state.user_email,
                "auth_required": st.session_state.auth_required,
                "has_auth_url": bool(st.session_state.auth_url),
                "conversation_id": st.session_state.conversation_id,
                "message_count": len(st.session_state.messages),
                "browser_session_id": st.session_state.get('browser_session_id', 'None')[:8] + "..."
            })

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