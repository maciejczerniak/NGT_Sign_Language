"""Unit tests for API Pydantic schemas."""

import pytest
from pydantic import ValidationError

from sign_language.api.schemas import (
    InfoResponse,
    PredictRequest,
    PredictResponse,
    ResetResponse,
    TopKItem,
)


# ---------------------------------------------------------------------------
# PredictRequest
# ---------------------------------------------------------------------------


class TestPredictRequest:
    def test_accepts_raw_base64(self):
        """Verify predict request accepts raw base64."""
        r = PredictRequest(image="abc123==")
        assert r.image == "abc123=="

    def test_accepts_data_url(self):
        """Verify predict request accepts data url."""
        r = PredictRequest(image="data:image/png;base64,abc123==")
        assert r.image.startswith("data:")

    def test_missing_image_raises(self):
        """Verify predict request missing image raises."""
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest()  # type: ignore[call-arg]
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("image",) for e in errors)

    def test_empty_string_is_accepted_by_pydantic(self):
        # Pydantic does not enforce non-empty; the preprocessing step rejects it
        """Verify predict request empty string is accepted by pydantic."""
        r = PredictRequest(image="")
        assert r.image == ""


# ---------------------------------------------------------------------------
# TopKItem
# ---------------------------------------------------------------------------


class TestTopKItem:
    def test_valid(self):
        """Verify top k item valid."""
        item = TopKItem(letter="A", confidence=0.9)
        assert item.letter == "A"
        assert item.confidence == pytest.approx(0.9)

    def test_confidence_zero(self):
        """Verify top k item confidence zero."""
        assert TopKItem(letter="Z", confidence=0.0).confidence == 0.0

    def test_confidence_one(self):
        """Verify top k item confidence one."""
        assert TopKItem(letter="Z", confidence=1.0).confidence == 1.0

    def test_missing_field_raises(self):
        """Verify top k item missing field raises."""
        with pytest.raises(ValidationError):
            TopKItem(letter="A")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# PredictResponse
# ---------------------------------------------------------------------------


class TestPredictResponse:
    _TOP_3 = [TopKItem(letter="A", confidence=0.9)]

    def _make(self, **overrides):
        defaults = dict(
            hand_detected=True,
            predicted_letter="A",
            confidence=0.9,
            top_3=self._TOP_3,
            stable_letter=None,
            stable_confidence=0.0,
            current_word="",
            sentence="",
            committed_letter=None,
        )
        return PredictResponse(**{**defaults, **overrides})

    def test_valid_full(self):
        """Verify predict response valid full."""
        r = self._make()
        assert r.predicted_letter == "A"
        assert r.hand_detected is True

    def test_optional_fields_accept_none(self):
        """Verify predict response optional fields accept none."""
        r = self._make(predicted_letter=None, stable_letter=None, committed_letter=None)
        assert r.predicted_letter is None
        assert r.stable_letter is None
        assert r.committed_letter is None

    def test_top_3_list_is_preserved(self):
        """Verify predict response top 3 list is preserved."""
        top = [
            TopKItem(letter=l, confidence=c)
            for l, c in [("A", 0.9), ("B", 0.07), ("C", 0.03)]
        ]
        r = self._make(top_3=top)
        assert len(r.top_3) == 3
        assert r.top_3[0].letter == "A"

    def test_no_hand_detected(self):
        """Verify predict response no hand detected."""
        r = self._make(hand_detected=False)
        assert r.hand_detected is False


# ---------------------------------------------------------------------------
# ResetResponse
# ---------------------------------------------------------------------------


class TestResetResponse:
    def test_default_ok_true(self):
        """Verify reset response default ok true."""
        assert ResetResponse().ok is True

    def test_can_override_to_false(self):
        """Verify reset response can override to false."""
        assert ResetResponse(ok=False).ok is False


# ---------------------------------------------------------------------------
# InfoResponse
# ---------------------------------------------------------------------------


class TestInfoResponse:
    def _make(self, **overrides):
        defaults = dict(
            app_name="SignTest",
            version="0.1.0",
            device="cpu",
            num_classes=26,
            class_names=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            landmark_model_available=False,
            hand_detector_available=False,
        )
        return InfoResponse(**{**defaults, **overrides})

    def test_valid(self):
        """Verify info response valid."""
        r = self._make()
        assert r.num_classes == 26
        assert len(r.class_names) == 26

    def test_landmark_flags(self):
        """Verify info response landmark flags."""
        r = self._make(landmark_model_available=True, hand_detector_available=True)
        assert r.landmark_model_available is True
        assert r.hand_detector_available is True

    def test_missing_required_field_raises(self):
        """Verify info response missing required field raises."""
        with pytest.raises(ValidationError):
            InfoResponse(  # type: ignore[call-arg]
                version="0.1.0",
                device="cpu",
                num_classes=3,
                class_names=["A"],
                landmark_model_available=False,
                hand_detector_available=False,
                # app_name intentionally omitted
            )
