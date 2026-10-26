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
