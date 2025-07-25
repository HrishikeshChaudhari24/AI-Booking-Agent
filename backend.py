import uvicorn
import os
from agent.controllers.booking_controller import app
FRONTEND_ORIGIN = "https://ai-booking-agent-efd.streamlit.app"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,          # <- mandatory
    allow_methods=["*"],
    allow_headers=["*"],
)
if __name__ == "__main__":
    # Host/port can be overridden by the deployment platform via the PORT env-var
    port = int(os.getenv("PORT", 10000))  # type: ignore  # noqa: PGH001
    uvicorn.run(app, host="0.0.0.0", port=port)
 
