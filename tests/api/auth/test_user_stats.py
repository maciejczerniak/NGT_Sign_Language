"""Tests for the per-user stats feature.

Covers:
  * ``level_band()`` — the pure level-band mapping (no DB).
  * ``GET  /api/stats``          — auth gating + response shape.
  * ``POST /api/stats/progress`` — points, streak, and letter progress;
    in particular that a *wrong* attempt does not advance the streak.
  * ``GET  /api/stats/letters``  — per-letter progress.
  * The daily-goal ``practiced_today`` count over distinct letters.

The auth fixtures (``client``, ``user_token``) come from
``tests/api/auth/conftest.py`` and are session-scoped, so the shared user's
points/streak accumulate across tests. Stateful assertions therefore check
*deltas* (values increased) rather than exact totals, and tests that need a
clean baseline register their own fresh user.
"""

import uuid

import pytest
from httpx import AsyncClient

from sign_language.api.schemas import level_band


# ─────────────────────────────────────────────────────────────────────
# level_band() — pure function, no DB
# ─────────────────────────────────────────────────────────────────────
class TestLevelBand:
    """The letters-learned → (level, name) mapping."""

    @pytest.mark.parametrize(
        "learned, expected",
        [
            (0, (1, "Beginner")),
            (1, (1, "Beginner")),
            (4, (1, "Beginner")),
            (5, (2, "Learner")),
            (9, (2, "Learner")),
            (10, (3, "Halfway")),
            (14, (3, "Halfway")),
            (15, (4, "Advanced")),
            (21, (4, "Advanced")),
            (22, (5, "Alphabet Master")),
        ],
    )
    def test_bands(self, learned, expected):
        """Each band boundary maps to the expected level + name."""
        assert level_band(learned) == expected

    def test_above_max_is_alphabet_master(self):
        """Counts above 22 (shouldn't happen) still resolve to the top band."""
        assert level_band(99) == (5, "Alphabet Master")

    def test_returns_tuple_of_int_and_str(self):
        """Return type is a (int, str) pair."""
        level, name = level_band(7)
        assert isinstance(level, int)
        assert isinstance(name, str)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
async def _register_and_login(client: AsyncClient) -> str:
    """Register a brand-new user and return its bearer token.

    Used by tests that need a clean stats baseline (streak 0, points 0)
    rather than the shared session-scoped user.
    """
    email = f"stats-{uuid.uuid4().hex[:12]}@example.com"
    password = "Str0ngPassword!"
    r = await client.post(
        "/api/auth/register", json={"email": email, "password": password}
    )
    assert r.status_code == 201, r.text
    r = await client.post(
        "/api/auth/jwt/login", data={"username": email, "password": password}
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ─────────────────────────────────────────────────────────────────────
# GET /api/stats
# ─────────────────────────────────────────────────────────────────────
class TestGetStats:
    async def test_requires_auth(self, client: AsyncClient):
        """Unauthenticated requests are rejected."""
        resp = await client.get("/api/stats")
        assert resp.status_code == 401

    async def test_fresh_user_defaults(self, client: AsyncClient):
        """A new user starts at zeroed stats, Beginner, with the expected shape."""
        token = await _register_and_login(client)
        resp = await client.get("/api/stats", headers=_auth(token))
        assert resp.status_code == 200, resp.text
        body = resp.json()

        for key in (
            "streak",
            "letters_learned",
            "total_letters",
            "points",
            "level",
            "level_name",
            "last_played",
            "practiced_today",
            "daily_goal",
        ):
            assert key in body, f"missing field: {key}"

        assert body["streak"] == 0
        assert body["letters_learned"] == 0
        assert body["points"] == 0
        assert body["level"] == 1
        assert body["level_name"] == "Beginner"
        assert body["last_played"] is None
        assert body["practiced_today"] == 0
        assert body["total_letters"] == 22


# ─────────────────────────────────────────────────────────────────────
# POST /api/stats/progress
# ─────────────────────────────────────────────────────────────────────
class TestReportProgress:
    async def test_requires_auth(self, client: AsyncClient):
        resp = await client.post(
            "/api/stats/progress",
            json={"letter": "A", "correct": True, "points": 10},
        )
        assert resp.status_code == 401

    async def test_correct_attempt_awards_points_and_starts_streak(
        self, client: AsyncClient
    ):
        """A first correct attempt adds points, sets streak to 1, and stores
        the activity name."""
        token = await _register_and_login(client)

        resp = await client.post(
            "/api/stats/progress",
            headers=_auth(token),
            json={
                "letter": "A",
                "correct": True,
                "points": 10,
                "activity": "Random Letters",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["points"] == 10
        assert body["streak"] == 1
        assert body["last_played"] == "Random Letters"
        assert body["letters_learned"] == 0

    async def test_incorrect_attempt_does_not_advance_streak(self, client: AsyncClient):
        """A wrong attempt must NOT advance the streak or award points —
        consistent with letter progress (regression guard for the streak fix)."""
        token = await _register_and_login(client)

        resp = await client.post(
            "/api/stats/progress",
            headers=_auth(token),
            json={"letter": "B", "correct": False, "points": 10},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["streak"] == 0, "wrong attempt should not advance the streak"
        assert body["points"] == 0, "wrong attempt should not award points"

    async def test_incorrect_then_correct_same_day(self, client: AsyncClient):
        """After a wrong attempt (streak stays 0), a correct attempt the same
        day brings the streak to 1."""
        token = await _register_and_login(client)

        await client.post(
            "/api/stats/progress",
            headers=_auth(token),
            json={"letter": "C", "correct": False, "points": 5},
        )
        resp = await client.post(
            "/api/stats/progress",
            headers=_auth(token),
            json={"letter": "C", "correct": True, "points": 5},
        )
        body = resp.json()
        assert body["streak"] == 1
        assert body["points"] == 5

    async def test_letter_becomes_learned_after_threshold(self, client: AsyncClient):
        """Signing the same letter correctly 3x marks it learned and bumps
        letters_learned to 1."""
        token = await _register_and_login(client)
        resp = None
        for _ in range(3):
            resp = await client.post(
                "/api/stats/progress",
                headers=_auth(token),
                json={"letter": "A", "correct": True, "points": 2},
            )
            assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["letters_learned"] == 1
        assert body["points"] == 6

    async def test_repeated_same_day_keeps_streak_at_one(self, client: AsyncClient):
        """Multiple correct attempts on the same day keep the streak at 1
        (it only increments across calendar days)."""
        token = await _register_and_login(client)
        last = None
        for _ in range(3):
            last = await client.post(
                "/api/stats/progress",
                headers=_auth(token),
                json={"letter": "D", "correct": True, "points": 1},
            )
        assert last.json()["streak"] == 1


# ─────────────────────────────────────────────────────────────────────
# Daily goal — practiced_today counts DISTINCT letters
# ─────────────────────────────────────────────────────────────────────
class TestDailyGoal:
    async def test_distinct_letters_counted_once(self, client: AsyncClient):
        """practiced_today counts distinct letters: A, B, A -> 2, not 3."""
        token = await _register_and_login(client)
        resp = None
        for letter in ("A", "B", "A"):
            resp = await client.post(
                "/api/stats/progress",
                headers=_auth(token),
                json={"letter": letter, "correct": True, "points": 1},
            )
            assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["practiced_today"] == 2

    async def test_daily_goal_value_present(self, client: AsyncClient):
        """The daily_goal target is reported (default 5)."""
        token = await _register_and_login(client)
        resp = await client.get("/api/stats", headers=_auth(token))
        assert resp.json()["daily_goal"] == 5


# ─────────────────────────────────────────────────────────────────────
# GET /api/stats/letters
# ─────────────────────────────────────────────────────────────────────
class TestGetLetterProgress:
    async def test_requires_auth(self, client: AsyncClient):
        resp = await client.get("/api/stats/letters")
        assert resp.status_code == 401

    async def test_shape_and_threshold(self, client: AsyncClient):
        """Response carries threshold, total_letters, and a letters list."""
        token = await _register_and_login(client)
        resp = await client.get("/api/stats/letters", headers=_auth(token))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "threshold" in body
        assert "total_letters" in body
        assert isinstance(body["letters"], list)
        assert body["total_letters"] == 22

    async def test_reflects_practice(self, client: AsyncClient):
        """After practising a letter, it appears with the right correct_count
        and learned flag."""
        token = await _register_and_login(client)
        for _ in range(3):
            await client.post(
                "/api/stats/progress",
                headers=_auth(token),
                json={"letter": "E", "correct": True, "points": 1},
            )
        resp = await client.get("/api/stats/letters", headers=_auth(token))
        letters = {item["letter"]: item for item in resp.json()["letters"]}
        assert "E" in letters
        assert letters["E"]["correct_count"] == 3
        assert letters["E"]["learned"] is True

    async def test_wrong_attempts_not_counted(self, client: AsyncClient):
        """Wrong attempts don't increment a letter's correct_count."""
        token = await _register_and_login(client)
        await client.post(
            "/api/stats/progress",
            headers=_auth(token),
            json={"letter": "F", "correct": False, "points": 1},
        )
        resp = await client.get("/api/stats/letters", headers=_auth(token))
        letters = {item["letter"]: item for item in resp.json()["letters"]}
        if "F" in letters:
            assert letters["F"]["correct_count"] == 0
            assert letters["F"]["learned"] is False
