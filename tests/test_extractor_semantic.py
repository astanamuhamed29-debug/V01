from core.pipeline.extractor_semantic import _extract_task_text


def test_extract_task_text_after_typo_imperative_and_url():
    text = "https://plati.market/itm/google-ai-pro-6-months-antigravity-gemini-3-1-nano-banana-pro-veo-3-1/5669298 слеоай кропию дизайна"
    assert _extract_task_text(text) == "кропию дизайна"
