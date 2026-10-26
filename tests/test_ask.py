from types import SimpleNamespace

import app.ask as ask_module


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        text="answer",
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=34,
        ),
    )
    models = SimpleNamespace(generate_content=lambda **kwargs: response)
    monkeypatch.setattr(
        ask_module.genai,
        "Client",
        lambda **kwargs: SimpleNamespace(models=models),
    )
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-api-key")

    assert ask_module.ask("question") == "answer"
    assert capsys.readouterr().out.strip() == "[トークン数] 入力: 12, 出力: 34"


def test_ask_reports_unavailable_token_usage(monkeypatch, capsys):
    response = SimpleNamespace(text="answer", usage_metadata=None)
    models = SimpleNamespace(generate_content=lambda **kwargs: response)
    monkeypatch.setattr(
        ask_module.genai,
        "Client",
        lambda **kwargs: SimpleNamespace(models=models),
    )
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-api-key")

    ask_module.ask("question")

    assert capsys.readouterr().out.strip() == "[トークン数] 利用不可"


def test_ask_interactively_reuses_chat_and_sends_each_question(monkeypatch, capsys):
    sent_questions = []
    responses = iter(
        [
            SimpleNamespace(
                text="first answer",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=12,
                    candidates_token_count=34,
                ),
            ),
            SimpleNamespace(
                text="second answer",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=56,
                    candidates_token_count=78,
                ),
            ),
        ]
    )
    chat = SimpleNamespace(
        send_message=lambda question: (
            sent_questions.append(question) or next(responses)
        )
    )
    created_chats = []
    chats = SimpleNamespace(
        create=lambda **kwargs: (created_chats.append(kwargs) or chat)
    )
    monkeypatch.setattr(
        ask_module.genai,
        "Client",
        lambda **kwargs: SimpleNamespace(chats=chats),
    )
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-api-key")
    questions = iter(["first question", "second question", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(questions))

    ask_module.ask_interactively()

    output = capsys.readouterr().out
    assert created_chats == [{"model": ask_module.MODEL}]
    assert sent_questions == ["first question", "second question"]
    assert "first answer" in output
    assert "second answer" in output
    assert "[トークン数] 入力: 12, 出力: 34" in output
    assert "[トークン数] 入力: 56, 出力: 78" in output
