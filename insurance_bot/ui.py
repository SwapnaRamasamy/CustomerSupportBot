"""Local browser interface for the existing grounded insurance chatbot."""
import logging
import time
import streamlit as st
from .config import Settings
from .graph import fallback
from .service import ChatService

logger = logging.getLogger(__name__)

SUGGESTIONS = [
    ("Claims", "How do I file an auto insurance claim?"),
    ("Coverage", "What is an auto insurance deductible?"),
    ("Billing", "My automatic premium payment failed. What should I do?"),
    ("Documents", "Where can I get proof of insurance?"),
]

STYLE = """<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {font-family:'DM Sans',sans-serif;}
[data-testid="stMainBlockContainer"] {max-width:960px;padding-top:2.6rem;padding-bottom:3rem;}
[data-testid="stSidebar"] {background:#102b43;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] label {color:#d5e3ee;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#b1c8d9;}
[data-testid="stSidebar"] button {border-color:#45627a;background:#193c57;color:#fff;border-radius:12px;}
[data-testid="stSidebar"] button:hover {border-color:#58d1c4;color:#9ff0e5;}
.brand {display:flex;align-items:center;gap:12px;margin:8px 0 42px;color:#fff;}
.brand-symbol {display:grid;place-items:center;width:44px;height:48px;border-radius:14px 14px 21px 21px;background:#1d958e;font-size:26px;}
.brand-name {font-family:'Manrope',sans-serif;font-size:23px;font-weight:800;letter-spacing:-.8px;}
.brand-sub {font-size:10px;letter-spacing:2px;color:#9fbdce;margin-top:4px;}
.sidebar-label {font-size:11px;letter-spacing:1.7px;color:#81a8c0;margin:28px 0 12px;}
.sidebar-note {font-size:13px;line-height:1.8;color:#b5cad8;}
.topline {display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:34px;}
.eyebrow {font-size:11px;letter-spacing:2px;font-weight:700;color:#688195;}
.context-badge {font-size:12px;color:#17665f;background:#e1f3ee;border:1px solid #c9e8df;border-radius:24px;padding:7px 13px;}
.hero h1 {font-family:'Manrope',sans-serif;font-size:clamp(32px,5vw,46px);line-height:1.16;letter-spacing:-1.8px;color:#142b43;margin:0 0 18px;font-weight:800;}
.hero h1 span {color:#087f8c;}
.hero p {font-size:16px;color:#61788b;line-height:1.7;max-width:620px;margin-bottom:28px;}
.section-label {font-size:11px;font-weight:700;letter-spacing:1.6px;color:#708496;margin:12px 0 10px;}
[data-testid="stMainBlockContainer"] [data-testid="stButton"] button {border:1px solid #dce5ed;background:#fff;color:#244259;border-radius:13px;padding:14px 18px;min-height:72px;text-align:left;justify-content:flex-start;box-shadow:0 3px 12px #102b4304;}
[data-testid="stMainBlockContainer"] [data-testid="stButton"] button:hover {border-color:#087f8c;background:#f0faf8;}
[data-testid="stChatMessage"] {border:1px solid #e1e8ef;border-radius:17px;background:#fff;margin:12px 0;padding:20px;box-shadow:0 3px 14px #142b4303;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {background:#edf6f6;border-color:#d3e8e7;}
[data-testid="stChatInput"] {border-radius:16px;border-color:#c8d9df;box-shadow:0 8px 28px #142b4309;}
[data-testid="stExpander"] {border-radius:10px;background:#f8fafc;}
.footnote {font-size:12px;color:#788b9b;line-height:1.6;margin:22px 0 6px;}
@media(max-width:640px) {[data-testid="stMainBlockContainer"]{padding-top:1.5rem;}.topline{margin-bottom:22px;}}
</style>"""

@st.cache_resource(show_spinner=False)
def get_service():
    # Shared stateless service; conversation content stays in per-session state.
    return ChatService()


def history_for_model(messages):
    return [{"role": m["role"], "content": m["content"]} for m in messages[-8:]]


def render_message(message):
    with st.chat_message(message["role"], avatar=":material/person:" if message["role"] == "user" else ":material/shield:"):
        if message["role"] == "assistant":
            st.caption("POLICY DESK")
        st.markdown(message["content"])
        result = message.get("result", {})
        if result.get("fallback"):
            st.info("Customer support can help with questions that require more information.", icon=":material/support_agent:")
        for number, source in enumerate(result.get("citations", []), 1):
            with st.expander(f"Source {number} | {source.get('title', 'Insurance document')}"):
                st.caption(f"{source.get('source', 'Document')} | {source.get('id', '')}")
                st.text(source.get("excerpt", ""))
        if message.get("elapsed") is not None:
            st.caption(f"Answered in {message['elapsed']:.0f}s | Based on your insurance documents")


def main():
    st.set_page_config(page_title="Policy Desk | Insurance Support", page_icon=":material/shield:", layout="centered")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.session_state.setdefault("messages", [])
    settings = Settings()
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-symbol">&#10003;</div><div><div class="brand-name">Policy desk</div><div class="brand-sub">CAR INSURANCE SUPPORT</div></div></div>', unsafe_allow_html=True)
        if st.button("New conversation", icon=":material/add:", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.markdown('<div class="sidebar-label">HERE TO HELP</div>', unsafe_allow_html=True)
        st.markdown("Claims & accident guidance\n\nCoverage & deductibles\n\nBilling & policy changes\n\nInsurance cards & documents")
        st.markdown('<div class="sidebar-label">ANSWERS YOU CAN CHECK</div><div class="sidebar-note">Responses use the provided insurance information. Open the sources beneath an answer to see the supporting text.</div>', unsafe_allow_html=True)
        st.divider()
        st.markdown("**Need a person?**")
        if settings.customer_support_phone.strip():
            st.text("Call " + settings.customer_support_phone.strip())
        else:
            st.caption("Contact your insurer or authorized agent. A customer support phone number has not been provided.")
        st.caption("Please use your insurer's secure channel for personal, policy, or payment details.")
    st.markdown('<div class="topline"><div class="eyebrow">YOUR INSURANCE HELP DESK</div><div class="context-badge">Answers with sources</div></div>', unsafe_allow_html=True)
    pending = None
    if not st.session_state.messages:
        st.markdown('<div class="hero"><h1>A little clarity.<br><span>A smoother road ahead.</span></h1><p>Make sense of your car insurance, one question at a time. Get guidance on claims, coverage, and the everyday details.</p></div><div class="section-label">WHERE WOULD YOU LIKE TO START?</div>', unsafe_allow_html=True)
        for start in range(0, len(SUGGESTIONS), 2):
            columns = st.columns(2)
            for column, (label, question) in zip(columns, SUGGESTIONS[start:start+2]):
                with column:
                    if st.button(f"{label} | {question}", key=f"suggest_{start}_{label}", use_container_width=True):
                        pending = question
        st.markdown('<div class="footnote">General guidance comes from the available documents. Coverage and account-specific decisions require confirmation from your insurer.</div>', unsafe_allow_html=True)
    else:
        st.title("Let's talk insurance")
        st.caption("Ask a follow-up anytime. Supporting sources appear beneath each answer.")
    for message in st.session_state.messages:
        render_message(message)
    question = st.chat_input("Ask about your car insurance...", max_chars=2000)
    question = (question or pending or "").strip()
    if not question:
        return
    history = history_for_model(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    render_message(st.session_state.messages[-1])
    started = time.perf_counter()
    with st.spinner("Checking your insurance documents... Local responses can take a minute or more."):
        try:
            result = get_service().ask(question, history)
        except Exception as exc:
            logger.error("UI service unavailable (%s)", type(exc).__name__)
            result = fallback(settings)
            result["answer"] = "The assistant is temporarily unavailable. Please try again shortly or contact customer support."
            if settings.customer_support_phone.strip():
                result["answer"] += " Call " + settings.customer_support_phone.strip() + "."
    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "result": result,
                                     "elapsed": time.perf_counter()-started})
    st.rerun()
