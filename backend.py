import uvicorn
import os
from agent.controllers.booking_controller import app

if __name__ == "__main__":
    # Host/port can be overridden by the deployment platform via the PORT env-var
    port = int(os.getenv("PORT", 8080))  # type: ignore  # noqa: PGH001
    uvicorn.run(app, host="0.0.0.0", port=port)
