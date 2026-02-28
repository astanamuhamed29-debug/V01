"""Tests for core/analytics/cognitive_detector.py."""

import pytest

from core.analytics.cognitive_detector import CognitiveDistortion, CognitiveDistortionDetector


@pytest.fixture
def detector() -> CognitiveDistortionDetector:
    return CognitiveDistortionDetector()


def test_detect_catastrophizing(detector):
    result = detector.detect("Это настоящая катастрофа, всё пропало!")
    types = [d.distortion_type for d in result]
    assert "CATASTROPHIZING" in types


def test_detect_black_white(detector):
    result = detector.detect("Я всегда всё делаю неправильно")
    types = [d.distortion_type for d in result]
    assert "BLACK_WHITE" in types


def test_detect_personalization(detector):
    result = detector.detect("Это произошло из-за меня, это моя вина")
    types = [d.distortion_type for d in result]
    assert "PERSONALIZATION" in types


def test_detect_overgeneralization(detector):
    result = detector.detect("Каждый раз одно и то же, постоянно так")
    types = [d.distortion_type for d in result]
    assert "OVERGENERALIZATION" in types


def test_detect_should_statements(detector):
    result = detector.detect("Я должен быть лучше, я обязан справляться")
    types = [d.distortion_type for d in result]
    assert "SHOULD_STATEMENTS" in types


def test_detect_labeling(detector):
    result = detector.detect("Я неудачник и я тупой")
    types = [d.distortion_type for d in result]
    assert "LABELING" in types


def test_detect_returns_distortion_dataclass(detector):
    result = detector.detect("Это катастрофа, всё пропало навсегда")
    assert len(result) > 0
    d = result[0]
    assert isinstance(d, CognitiveDistortion)
    assert d.distortion_type
    assert 0.0 <= d.confidence <= 1.0
    assert d.evidence_text
    assert d.reframe_suggestion


def test_detect_no_distortions_on_neutral_text(detector):
    result = detector.detect("Сегодня хорошая погода и я пошёл гулять.")
    # Neutral text should produce few or zero distortions
    assert len(result) == 0


def test_detect_multiple_distortions(detector):
    text = "Я всегда всё порчу — это катастрофа и это только моя вина."
    result = detector.detect(text)
    assert len(result) >= 2


def test_detect_confidence_between_0_and_1(detector):
    result = detector.detect("Я должен быть лучше, я обязан, никогда не справляюсь")
    for d in result:
        assert 0.0 <= d.confidence <= 1.0
