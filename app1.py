# FINAL WORKING LangGraph + SQLite persistent multi-chat chatbot
# Full thread visible, no disappearing messages, stable

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
import streamlit as st
import time
import uuid
import os
from dotenv import load_dotenv

load_dotenv()
import json

# Load saved titles on startup
TITLES_FILE = "chat_titles.json"
if os.path.exists(TITLES_FILE):
    with open(TITLES_FILE, "r") as f:
        st.session_state.chat_titles = json.load(f)
else:
    st.session_state.chat_titles = {}

# ---------------- LANGGRAPH PERSISTENT SETUP ----------------
conn = SqliteSaver.from_conn_string("chat_history.db")
memory = conn.__enter__()

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

graph_builder = StateGraph(State)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=512,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def chatbot(state: State):
    messages = state["messages"]
    if messages and not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content="You are a helpful AI assistant.")] + messages
    response = llm.invoke(messages)
    return {"messages": [response]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile(checkpointer=memory)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("Chatbot 🔥")

# UI state
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = {}

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("About")
    st.write("Persistent chatbot using **LangGraph + SQLite**")
    st.caption("Model: llama-3.3-70b-versatile")
    st.divider()
    st.subheader("Chat History")

    if st.button("✨ New Chat", type="primary"):
        new_id = str(uuid.uuid4())
        st.session_state.current_chat_id = new_id
        st.session_state.chat_titles[new_id] = "New Chat"
        st.rerun()

    all_checkpoints = list(memory.list(config=None))
    threads = {}
    for cp in all_checkpoints:
        thread_id = cp.config["configurable"]["thread_id"]
        ts = cp.checkpoint["ts"]
        if thread_id not in threads or ts > threads[thread_id].checkpoint["ts"]:
            threads[thread_id] = cp

    sorted_threads = sorted(threads.values(), key=lambda x: x.checkpoint["ts"], reverse=True)

    for cp in sorted_threads:
        thread_id = cp.config["configurable"]["thread_id"]
        title = st.session_state.chat_titles.get(thread_id, "New Chat")
        if st.button(f"📄 {title}", key=thread_id):
            st.session_state.current_chat_id = thread_id
            st.rerun()

# ---------------- CURRENT THREAD ----------------
if st.session_state.current_chat_id is None:
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.chat_titles[st.session_state.current_chat_id] = "New Chat"

thread_id = st.session_state.current_chat_id
config = {"configurable": {"thread_id": thread_id}}

# ---------------- LOAD LATEST MESSAGES ----------------
# ---------------- LOAD LATEST MESSAGES (RELIABLE) ----------------
current_messages = []

# Get the current state directly — this is the most reliable way
state = graph.get_state(config)
if state and state.values and "messages" in state.values:
    current_messages = state.values["messages"]

# ---------------- DISPLAY FULL THREAD ----------------
for message in current_messages:
    if isinstance(message, SystemMessage):
        continue
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# ---------------- USER INPUT ----------------
user_input = st.chat_input("Type your message here...", key="main_chat_input")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Show thinking spinner and run the graph (synchronous, ensures save)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )

    # Auto-update title
    if st.session_state.chat_titles.get(thread_id, "New Chat") == "New Chat":
        short_title = user_input[:30] + ("..." if len(user_input) > 30 else "")
        st.session_state.chat_titles[thread_id] = short_title

    # Save to file so it survives restart
    with open(TITLES_FILE, "w") as f:
        json.dump(st.session_state.chat_titles, f)
    # Now safe to rerun — new message is fully saved
    st.rerun()