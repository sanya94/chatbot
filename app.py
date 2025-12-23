# END-TO-END PERSONAL MULTI-USER CHATBOT
# Private per-browser threads via localStorage
# Persistent titles in chat_titles.json (with safety cleaning)

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
import uuid
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

# ────────────── PERSISTENT TITLES (JSON file) ──────────────
TITLES_FILE = "chat_titles.json"

if "chat_titles" not in st.session_state:
    if os.path.exists(TITLES_FILE):
        try:
            with open(TITLES_FILE, "r") as f:
                st.session_state.chat_titles = json.load(f)
            # Safety: remove any invalid keys (like DeltaGenerator)
            st.session_state.chat_titles = {
                k: v for k, v in st.session_state.chat_titles.items()
                if isinstance(k, (str, int, float, bool, type(None)))
            }
        except (json.JSONDecodeError, IOError):
            st.session_state.chat_titles = {}
    else:
        st.session_state.chat_titles = {}

def save_titles():
    # Safety: only save valid keys
    clean_titles = {
        k: v for k, v in st.session_state.chat_titles.items()
        if isinstance(k, (str, int, float, bool, type(None)))
    }
    with open(TITLES_FILE, "w") as f:
        json.dump(clean_titles, f)

# ────────────── UNIQUE THREAD ID PER BROWSER (localStorage) ──────────────
if "thread_id" not in st.session_state:
    js_code = """
    if (!localStorage.getItem('chatbot_thread_id')) {
        localStorage.setItem('chatbot_thread_id', 'thread_' + crypto.randomUUID());
    }
    parent.window.streamlit.setComponentValue(localStorage.getItem('chatbot_thread_id'));
    """
    component_value = st.components.v1.html(
        f"<script>{js_code}</script>",
        height=0
    )
    st.session_state.thread_id = component_value if component_value else str(uuid.uuid4())

thread_id = st.session_state.thread_id

# ────────────── LANGGRAPH + SQLITE SETUP ──────────────
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

# ────────────── STREAMLIT UI ──────────────
st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("Chatbot 🔥 (CI/CD Test)")

# ────────────── SIDEBAR ──────────────
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
        save_titles()
        st.rerun()

    all_checkpoints = list(memory.list(config=None))
    threads = {}
    for cp in all_checkpoints:
        tid = cp.config["configurable"]["thread_id"]
        ts = cp.checkpoint["ts"]
        if tid not in threads or ts > threads[tid].checkpoint["ts"]:
            threads[tid] = cp

    sorted_threads = sorted(threads.values(), key=lambda x: x.checkpoint["ts"], reverse=True)

    for cp in sorted_threads:
        tid = cp.config["configurable"]["thread_id"]
        title = st.session_state.chat_titles.get(tid, "New Chat")
        if st.button(f"📄 {title}", key=tid):
            st.session_state.current_chat_id = tid
            st.rerun()

# ────────────── CURRENT THREAD ──────────────
if "current_chat_id" not in st.session_state or st.session_state.current_chat_id is None:
    st.session_state.current_chat_id = thread_id
    if thread_id not in st.session_state.chat_titles:
        st.session_state.chat_titles[thread_id] = "New Chat"
        save_titles()

current_thread_id = st.session_state.current_chat_id
config = {"configurable": {"thread_id": current_thread_id}}

# ────────────── LOAD MESSAGES ──────────────
current_messages = []
state = graph.get_state(config)
if state and state.values and "messages" in state.values:
    current_messages = state.values["messages"]

# ────────────── DISPLAY FULL THREAD ──────────────
for message in current_messages:
    if isinstance(message, SystemMessage):
        continue
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# ────────────── USER INPUT ──────────────
user_input = st.chat_input("Type your message here...", key="main_chat_input")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run graph with thinking indicator
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )

    # Auto-update title (only for first message)
    if st.session_state.chat_titles.get(current_thread_id, "New Chat") == "New Chat":
        short_title = user_input[:30] + ("..." if len(user_input) > 30 else "")
        st.session_state.chat_titles[current_thread_id] = short_title
        save_titles()

    st.rerun()