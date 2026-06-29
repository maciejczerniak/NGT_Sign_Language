from pathlib import Path

from sign_language_training.run_naming import (
    _find_existing_run_numbers,
    generate_run_name,
    resolve_run_dir,
)


class TestFindExistingRunNumbers:
    """Tests for the ``_find_existing_run_numbers`` helper."""

    def test_returns_empty_list_when_directory_does_not_exist(
        self, tmp_path: Path
    ) -> None:
        """Return an empty list when the output directory does not exist."""
        missing = tmp_path / "nonexistent"
        assert _find_existing_run_numbers(missing) == []

    def test_returns_empty_list_when_directory_is_empty(self, tmp_path: Path) -> None:
        """Return an empty list when the output directory has no subfolders."""
        assert _find_existing_run_numbers(tmp_path) == []

    def test_returns_empty_list_when_no_matching_folders_exist(
        self, tmp_path: Path
    ) -> None:
        """Return an empty list when no folders match the run naming pattern."""
        (tmp_path / "unrelated_folder").mkdir()
        (tmp_path / "some_file.txt").write_text("hello")
        assert _find_existing_run_numbers(tmp_path) == []

    def test_finds_single_run(self, tmp_path: Path) -> None:
        """Detect a single existing run folder and return its number."""
        (tmp_path / "model_1").mkdir()
        assert _find_existing_run_numbers(tmp_path) == [1]

    def test_finds_multiple_runs_sorted(self, tmp_path: Path) -> None:
        """Detect multiple run folders and return their numbers in sorted order."""
        (tmp_path / "model_3").mkdir()
        (tmp_path / "model_1").mkdir()
        (tmp_path / "model_10").mkdir()
        assert _find_existing_run_numbers(tmp_path) == [1, 3, 10]

    def test_ignores_files_with_matching_name(self, tmp_path: Path) -> None:
        """Ignore regular files whose names match the run pattern."""
        (tmp_path / "model_1").write_text("not a directory")
        (tmp_path / "model_2").mkdir()
        assert _find_existing_run_numbers(tmp_path) == [2]

    def test_ignores_folders_with_wrong_prefix(self, tmp_path: Path) -> None:
        """Ignore folders that do not match the expected naming prefix."""
        (tmp_path / "model_1").mkdir()
        (tmp_path / "other_model_2").mkdir()
        (tmp_path / "model_extra").mkdir()
        assert _find_existing_run_numbers(tmp_path) == [1]


class TestGenerateRunName:
    """Tests for the ``generate_run_name`` function."""

    def test_returns_first_run_for_empty_directory(self, tmp_path: Path) -> None:
        """Return ``model_1`` when the output directory is empty."""
        assert generate_run_name(tmp_path) == "model_1"

    def test_returns_first_run_when_directory_does_not_exist(
        self, tmp_path: Path
    ) -> None:
        """Return ``model_1`` when the output directory does not exist."""
        missing = tmp_path / "nonexistent"
        assert generate_run_name(missing) == "model_1"

    def test_returns_next_number_after_existing_runs(self, tmp_path: Path) -> None:
        """Return the next sequential number after the highest existing run."""
        (tmp_path / "model_1").mkdir()
        (tmp_path / "model_2").mkdir()
        assert generate_run_name(tmp_path) == "model_3"

    def test_returns_next_after_highest_with_gaps(self, tmp_path: Path) -> None:
        """Return the next number after the highest even when there are gaps."""
        (tmp_path / "model_1").mkdir()
        (tmp_path / "model_5").mkdir()
        assert generate_run_name(tmp_path) == "model_6"


class TestResolveRunDir:
    """Tests for the ``resolve_run_dir`` function."""

    def test_auto_generates_name_when_none(self, tmp_path: Path) -> None:
        """Auto-generate the next run name when ``run_name`` is ``None``."""
        (tmp_path / "model_1").mkdir()
        result = resolve_run_dir(tmp_path, run_name=None)
        assert result == tmp_path / "model_2"

    def test_uses_explicit_name_when_provided(self, tmp_path: Path) -> None:
        """Use the provided run name directly without auto-generating."""
        result = resolve_run_dir(tmp_path, run_name="my_custom_run")
        assert result == tmp_path / "my_custom_run"

    def test_does_not_create_directory(self, tmp_path: Path) -> None:
        """Return a path without creating the directory on disk."""
        result = resolve_run_dir(tmp_path, run_name=None)
        assert not result.exists()
