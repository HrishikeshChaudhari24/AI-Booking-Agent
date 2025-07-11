

<h1 align="center">AI-Powered Booking Agent</h1>

<p align="center">
  A sophisticated, conversational AI agent that seamlessly books appointments in your Google Calendar.
  <br />
  <a href="https://ai-booking-agent-efd.streamlit.app/"><strong>View Live Demo »</strong></a>
  <br />
  <br />
  <a href="https://github.com/Tinum-Contos/AI-Booking-Agent/issues">Report Bug</a>
  ·
  <a href="https://github.com/Tinum-Contos/AI-Booking-Agent/issues">Request Feature</a>
</p>

<!-- Shields -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-FastAPI-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Frontend-Streamlit-orange.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/LLM-Gemini_&_Groq-purple.svg" alt="LLM">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

---

## 🌟 About The Project

This project implements a powerful AI-driven booking agent. Users can interact with it in a natural, conversational way to check availability and schedule appointments directly into their Google Calendar. The agent is built with a robust FastAPI backend and an intuitive Streamlit frontend, leveraging state-of-the-art language models from Google (Gemini) and Groq (Llama 3) for intent recognition and response generation.

**Live Application:** [https://ai-booking-agent-efd.streamlit.app/](https://ai-booking-agent-efd.streamlit.app/)

### ✨ Key Features

*   **Conversational Interface**: Chat with the agent using natural language.
*   **Google Calendar Integration**: Fetches your availability and books appointments in real-time.
*   **Secure Authentication**: Uses Google OAuth2 to securely connect to your calendar.
*   **Multi-LLM Support**: Powered by Google's Gemini and Groq's Llama 3 for fast and accurate responses.
*   **Stateful Conversations**: Remembers the context of your conversation for a smooth user experience, powered by LangGraph.
*   **Persistent Sessions**: User sessions are maintained across browser restarts.

---

## 🛠️ Built With

This project is built with a modern stack, combining powerful backend and frontend technologies with cutting-edge AI models.

**Frontend:**
*   <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">

**Backend:**
*   <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
*   <img src="https://img.shields.io/badge/Uvicorn-009688?style=for-the-badge&logo=python&logoColor=white" alt="Uvicorn">

**AI & Language Models:**
*   <img src="https://img.shields.io/badge/Google_Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Google Gemini">
*   <img src="https://img.shields.io/badge/Groq-00B592?style=for-the-badge&logo=groq&logoColor=white" alt="Groq">
*   <img src="https://img.shields.io/badge/LangGraph-FF69B4?style=for-the-badge" alt="LangGraph">

**Database & Session Management:**
*   <img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite">
*   <img src="https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white" alt="Redis">


---

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

*   Python 3.11+
*   `pip` for package management

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/Tinum-Contos/AI-Booking-Agent.git
    cd AI-Booking-Agent
    ```

2.  **Create a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root of the project and add the following variables. You'll need to get your own API keys from Google and Groq.

    ```env
    # Google API Credentials
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
    GOOGLE_CREDENTIALS_JSON='{"web":{"client_id":"...", "project_id":"...", ...}}'

    # Groq API Key
    GROQ_API_KEY="YOUR_GROQ_API_KEY"
    ```

    *   You can get `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   `GOOGLE_CREDENTIALS_JSON` is the JSON file you download from Google Cloud Console when you create OAuth 2.0 Client IDs. You need to paste the content of the JSON file as a single line string.
    *   You can get `GROQ_API_KEY` from [GroqCloud](https://console.groq.com/keys).

### Running the Application

Once the setup is complete, you can run the application with a single command:

```sh
python main.py
```

This will start the FastAPI backend and then launch the Streamlit frontend in your browser.

---

## 📁 Project Structure

<pre>
AI-Booking-Agent/
├── agent/
│   ├── controllers/
│   │   └── booking_controller.py  # FastAPI app, agent logic, LLM calls
│   ├── models/
│   │   ├── calendar.py          # Google Calendar functions
│   │   └── database.py          # Database interactions (SQLite, Redis)
│   ├── views/
│   │   └── booking_view.py      # Streamlit UI
│   └── utils/
│       └── ...                  # Utility functions
├── .env.example                 # Example environment variables
├── main.py                      # Main application entry point
├── backend.py                   # Secondary entrypoint for backend
├── requirements.txt             # Project dependencies
└── README.md                    # This file
</pre>

---

## 🚢 Deployment

This application is deployed on Streamlit Community Cloud. You can access the live version here:

[https://ai-booking-agent-efd.streamlit.app/](https://ai-booking-agent-efd.streamlit.app/)

---

## 🤝 Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📧 Contact


- **Name:** Hrishikesh Chaudhari
- **Email:** [chaudharihrishikesh30@gmail.com](mailto:chaudharihrishikesh30@gmail.com)
- **LinkedIn:** [https://www.linkedin.com/in/hrishikesh-chaudhari-169308248/](https://www.linkedin.com/in/hrishikesh-chaudhari-169308248/)
</div> 
