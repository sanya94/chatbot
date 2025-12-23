# Chatbot 🤖

A persistent, multi-session chatbot built with **Streamlit**, **LangGraph**, and **Groq**. This application supports maintaining chat history across sessions using SQLite and allows for multiple concurrent chat threads.

## Features

- **Persistent Chat History**: Uses SQLite to save chat history, so your conversations are not lost when you refresh the page.
- **Multiple Chat Threads**: Create and switch between multiple independent chat sessions.
- **Auto-Titling**: Automatically generates titles for new chats based on the first message.
- **Powered by Groq**: Utilizes the `llama-3.3-70b-versatile` model for fast and intelligent responses.

## Prerequisites

- Python 3.9 or higher
- A Groq API Key

## Installation

1.  **Clone the repository** (or download the files):
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables**:
    Create a `.env` file in the root directory and add your Groq API key:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

## Project Structure

- `app.py`: Main application code containing the Streamlit UI and LangGraph logic.
- `requirements.txt`: List of Python dependencies.
- `chat_history.db`: SQLite database file (created automatically) storing chat history.
- `chat_titles.json`: JSON file (created automatically) storing titles for chat threads.
