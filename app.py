import os
import sys
from typing import Any, Dict, List

# Ensure local src/ is importable without editable install
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

import asyncio
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage, BaseMessage

from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.frontend.storage import (
    list_sessions,
    new_session,
    load_session,
    save_session,
    rename_session,
    delete_session,
)


# -------- Helpers --------
def to_lc_messages(msgs: List[Dict[str, str]]) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    for m in msgs:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        # ignore others for now
    return out


def to_simple_messages(lc_msgs: List[BaseMessage]) -> List[Dict[str, str]]:
    simple: List[Dict[str, str]] = []
    for m in lc_msgs:
        if isinstance(m, HumanMessage):
            simple.append({"role": "user", "content": str(m.content)})
        elif isinstance(m, AIMessage):
            simple.append({"role": "assistant", "content": str(m.content)})
        # Skip Tool/System messages from chat view
    return simple


def render_messages(simple_msgs: List[Dict[str, str]]) -> None:
    for m in simple_msgs:
        if m["role"] not in ("user", "assistant"):
            continue
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["content"])  # markdown render


def ensure_session_state_defaults() -> None:
    st.session_state.setdefault("current_session_id", None)
    st.session_state.setdefault("session_cache", {})  # id -> session dict


def run_async(coro):
    """Run an async coroutine from Streamlit, handling existing event loops."""
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
        raise


async def run_graph_with_events(lc_messages: List[BaseMessage], stage_container) -> Dict[str, Any]:
    """Run the graph and stream node-level updates as they complete."""
    placeholder = stage_container.empty()
    md_parts: List[str] = ["### Live steps"]

    def flush():
        placeholder.markdown("\n\n".join(md_parts))

    seen_raw_notes: List[str] = []
    seen_notes: List[str] = []
    research_brief_shown = False
    final_report_shown = False
    final_state: Dict[str, Any] | None = None

    async for event in deep_researcher.astream_events({"messages": lc_messages}, version="v2"):
        etype = event.get("event")
        name = event.get("name")
        data = event.get("data", {}) or {}

        if etype == "on_node_start":
            md_parts.append(f"▶️ Starting: `{name}`")
            flush()
            continue

        if etype == "on_node_end":
            out = data.get("output", {}) or {}
            if hasattr(out, "dict"):
                try:
                    out = out.dict()
                except Exception:
                    out = {}

            if not research_brief_shown and isinstance(out, dict) and out.get("research_brief"):
                md_parts.append("#### 🧭 Research Brief")
                md_parts.append(str(out.get("research_brief")))
                research_brief_shown = True

            if isinstance(out, dict) and out.get("raw_notes"):
                raw_notes = out.get("raw_notes") or []
                if isinstance(raw_notes, list):
                    new_items = [n for n in raw_notes if n not in seen_raw_notes]
                    if new_items:
                        md_parts.append("#### 📝 New Raw Notes")
                        md_parts.extend([f"- {n}" for n in new_items])
                        seen_raw_notes.extend(new_items)

            # Individual researcher compressed outputs
            if isinstance(out, dict) and out.get("compressed_research"):
                md_parts.append("#### 🔎 Research Synthesis")
                md_parts.append(str(out.get("compressed_research")))

            if isinstance(out, dict) and out.get("notes"):
                notes = out.get("notes") or []
                if isinstance(notes, list):
                    new_items = [n for n in notes if n not in seen_notes]
                    if new_items:
                        md_parts.append("#### 📌 New Notes")
                        md_parts.extend([f"- {n}" for n in new_items])
                        seen_notes.extend(new_items)

            if not final_report_shown and isinstance(out, dict) and out.get("final_report"):
                md_parts.append("#### 📄 Final Report (preview)")
                md_parts.append(str(out.get("final_report")))
                final_report_shown = True

            flush()
            continue

        if etype == "on_chain_end":
            final_state = data.get("output")
            flush()

    return final_state or {}


# -------- UI --------
st.set_page_config(page_title="Open Deep Research - Chat", layout="wide")
ensure_session_state_defaults()

# Load environment variables from .env (for API keys)
load_dotenv(find_dotenv(), override=False)

# Sidebar: sessions
st.sidebar.title("Sessions")
sessions = list_sessions()

if st.sidebar.button("➕ New Chat", use_container_width=True):
    s = new_session()
    st.session_state.current_session_id = s["id"]
    st.session_state.session_cache[s["id"]] = s
    st.rerun()

for s in sessions:
    selected = st.session_state.current_session_id == s["id"]
    cols = st.sidebar.columns([1, 0.3])
    if cols[0].button(("🗨️ " if selected else "") + s["title"], key=f"sel_{s['id']}", use_container_width=True):
        st.session_state.current_session_id = s["id"]
        sess = load_session(s["id"]) or s
        st.session_state.session_cache[s["id"]] = sess
        st.rerun()
    if cols[1].button("🗑️", key=f"del_{s['id']}"):
        delete_session(s["id"])
        if st.session_state.current_session_id == s["id"]:
            st.session_state.current_session_id = None
        st.rerun()

st.sidebar.markdown("---")
if st.session_state.current_session_id:
    # Rename control
    sess = st.session_state.session_cache.get(st.session_state.current_session_id) or load_session(st.session_state.current_session_id)
    if sess:
        new_title = st.sidebar.text_input("Rename session", value=sess.get("title", "Untitled"))
        if st.sidebar.button("Rename"):
            rename_session(sess["id"], new_title)
            sess["title"] = new_title
            st.session_state.session_cache[sess["id"]] = sess
            st.rerun()


# Main area
st.title("🔬 Open Deep Research — Chat Prototype")

if not st.session_state.current_session_id:
    st.info("Create or select a chat in the sidebar to begin.")
    st.stop()

session = st.session_state.session_cache.get(st.session_state.current_session_id)
if not session:
    session = load_session(st.session_state.current_session_id)
    if not session:
        st.error("Session not found.")
        st.stop()
    st.session_state.session_cache[session["id"]] = session


# Chat transcript
render_messages(session.get("messages", []))


# Chat input
prompt = st.chat_input("Ask a research question…")
if prompt:
    # Append user message
    session["messages"].append({"role": "user", "content": prompt})
    if not session.get("title") or session["title"] == "New Chat":
        session["title"] = prompt.strip()[:80]

    # Show immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare invocation
    lc_messages = to_lc_messages(session["messages"])  # full context
    previous_len = len(session["messages"])

    live = st.expander("Live steps", expanded=True)
    try:
        result = run_async(run_graph_with_events(lc_messages, live))
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error: {e}")
        save_session(session["id"], session)
        st.stop()

    # Extract new messages from result and render
    all_msgs_simple = to_simple_messages(result.get("messages", []))
    new_msgs = all_msgs_simple[previous_len:]
    for m in new_msgs:
        with st.chat_message("assistant"):
            st.markdown(m["content"])
    
    # Persist interim artifacts
    session["messages"] = all_msgs_simple
    session["research_brief"] = result.get("research_brief")
    session["notes"] = result.get("notes", [])
    session["raw_notes"] = result.get("raw_notes", [])
    session["final_report"] = result.get("final_report")
    save_session(session["id"], session)


# Sidebar (right column style) for interim artifacts
st.markdown("---")
cols = st.columns(3)
with cols[0]:
    st.subheader("Research Brief")
    rb = session.get("research_brief")
    st.write(rb or "—")
with cols[1]:
    st.subheader("Notes")
    notes = session.get("notes", [])
    if notes:
        for n in notes:
            st.markdown(f"- {n}")
    else:
        st.write("—")
with cols[2]:
    st.subheader("Raw Notes")
    rnotes = session.get("raw_notes", [])
    if rnotes:
        for n in rnotes:
            st.markdown(f"- {n}")
    else:
        st.write("—")


# Final report section
st.markdown("---")
st.subheader("Final Report")
fr = session.get("final_report")
if fr:
    st.markdown(fr)
else:
    st.caption("No final report yet. Some interactions may end with a clarifying question first.")
