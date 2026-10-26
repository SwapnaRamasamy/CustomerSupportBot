from pathlib import Path
from streamlit.testing.v1 import AppTest
from insurance_bot import ui

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"

class FakeService:
    def __init__(self):
        self.calls=[]
    def ask(self, question, history):
        self.calls.append((question,history))
        return {"answer":"Check your declarations page. [faq:0]", "fallback":False,
                "citations":[{"id":"faq:0","title":"Deductibles","source":"FAQ","excerpt":"Your deductible is on your declarations page."}]}


def test_chat_sources_followup_and_reset(monkeypatch):
    service=FakeService()
    monkeypatch.setattr(ui,"get_service",lambda:service)
    app=AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert len(app.button)==5
    app.chat_input[0].set_value("What is my deductible?").run(timeout=20)
    assert not app.exception
    assert len(app.chat_message)==2
    assert len(app.expander)==1
    assert "Deductibles" in app.expander[0].label
    assert service.calls[0][1]==[]
    app.chat_input[0].set_value("Where can I find it?").run(timeout=20)
    assert len(service.calls[1][1])==2
    assert service.calls[1][1][0]["content"]=="What is my deductible?"
    app.button[0].click().run(timeout=20)
    assert app.session_state["messages"]==[]
    assert len(app.chat_message)==0


def test_suggestion_submits_question(monkeypatch):
    service=FakeService()
    monkeypatch.setattr(ui,"get_service",lambda:service)
    app=AppTest.from_file(str(APP)).run(timeout=20)
    app.button(key="suggest_0_Claims").click().run(timeout=20)
    assert not app.exception
    assert service.calls[0][0]==ui.SUGGESTIONS[0][1]
    assert len(app.chat_message)==2


def test_service_failure_is_safe_and_sessions_separate(monkeypatch):
    def unavailable():
        raise RuntimeError("secret provider payload should never be shown")
    monkeypatch.setattr(ui,"get_service",unavailable)
    app=AppTest.from_file(str(APP)).run(timeout=20)
    app.chat_input[0].set_value("Help with a claim").run(timeout=20)
    assert not app.exception
    messages=app.session_state["messages"]
    assert messages[-1]["result"]["fallback"]
    assert "temporarily unavailable" in messages[-1]["content"]
    assert "secret provider" not in messages[-1]["content"]
    assert len(app.info)==1
    second=AppTest.from_file(str(APP)).run(timeout=20)
    assert second.session_state["messages"]==[]
