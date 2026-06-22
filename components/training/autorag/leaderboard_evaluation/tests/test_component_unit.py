"""Tests for the leaderboard_evaluation component."""

import json
from pathlib import Path
from unittest import mock

import pytest

from ..component import leaderboard_evaluation


class TestLeaderboardEvaluationUnitTests:
    """Unit tests for leaderboard_evaluation success and failure paths."""

    def test_component_function_exists(self):
        """Component factory exists and exposes python_func."""
        assert callable(leaderboard_evaluation)
        assert hasattr(leaderboard_evaluation, "python_func")

    def test_missing_rag_patterns_dir_raises_file_not_found(self, tmp_path):
        """Non-existing rag_patterns input path raises FileNotFoundError."""
        html_artifact = mock.MagicMock(path=str(tmp_path / "out.html"))
        with pytest.raises(FileNotFoundError, match="rag_patterns path is not a directory"):
            leaderboard_evaluation.python_func(
                rag_patterns=str(tmp_path / "missing"),
                html_artifact=html_artifact,
                optimization_metric="faithfulness",
            )

    def test_generates_html_from_pattern_json(self, tmp_path):
        """Valid pattern directory produces non-empty HTML leaderboard."""
        rag_patterns_dir = tmp_path / "patterns"
        pattern_dir = rag_patterns_dir / "pattern_a"
        pattern_dir.mkdir(parents=True)
        (pattern_dir / "pattern.json").write_text(
            json.dumps(
                {
                    "name": "pattern_a",
                    "scores": {
                        "faithfulness": {"mean": 0.95},
                        "answer_correctness": {"mean": 0.83},
                    },
                    "settings": {
                        "chunking": {"method": "recursive", "chunk_size": 512, "chunk_overlap": 64},
                        "embedding_model": "embed-1",
                        "retrieval": {"method": "vector", "number_of_chunks": 5, "search_mode": "vector"},
                        "foundation_model": "gen-1",
                    },
                }
            ),
            encoding="utf-8",
        )
        html_artifact = mock.MagicMock(path=str(tmp_path / "leaderboard.html"))

        leaderboard_evaluation.python_func(
            rag_patterns=str(rag_patterns_dir),
            html_artifact=html_artifact,
            optimization_metric="faithfulness",
        )

        html_text = Path(html_artifact.path).read_text(encoding="utf-8")
        assert "RAG Patterns Leaderboard" in html_text
        assert "pattern_a" in html_text
        assert "faithfulness" in html_text

    def test_ranks_patterns_by_final_score_descending(self, tmp_path):
        """Higher final_score patterns appear first and are marked as best."""
        rag_patterns_dir = tmp_path / "patterns"
        for name, score in (("pattern_low", 0.72), ("pattern_high", 0.95)):
            pattern_dir = rag_patterns_dir / name
            pattern_dir.mkdir(parents=True)
            (pattern_dir / "pattern.json").write_text(
                json.dumps(
                    {
                        "name": name,
                        "final_score": score,
                        "scores": {"faithfulness": {"mean": score}},
                        "settings": {
                            "chunking": {"method": "recursive", "chunk_size": 512, "chunk_overlap": 64},
                            "embedding": {"model_id": "embed-1"},
                            "retrieval": {"method": "vector", "number_of_chunks": 5},
                            "generation": {"model_id": "gen-1"},
                        },
                    }
                ),
                encoding="utf-8",
            )

        html_artifact = mock.MagicMock(path=str(tmp_path / "leaderboard.html"))
        leaderboard_evaluation.python_func(
            rag_patterns=str(rag_patterns_dir),
            html_artifact=html_artifact,
            optimization_metric="faithfulness",
        )

        html_text = Path(html_artifact.path).read_text(encoding="utf-8")
        assert html_text.index("pattern_high") < html_text.index("pattern_low")
        assert "Best pattern: <strong>pattern_high</strong>" in html_text
        assert 'class="rank-1"' in html_text

    def test_optimization_metric_column_prioritized_in_header(self, tmp_path):
        """Leaderboard highlights the configured optimization metric column."""
        rag_patterns_dir = tmp_path / "patterns"
        pattern_dir = rag_patterns_dir / "pattern_a"
        pattern_dir.mkdir(parents=True)
        (pattern_dir / "pattern.json").write_text(
            json.dumps(
                {
                    "name": "pattern_a",
                    "final_score": 0.88,
                    "scores": {
                        "faithfulness": {"mean": 0.88},
                        "answer_correctness": {"mean": 0.77},
                    },
                    "settings": {
                        "chunking": {"method": "recursive", "chunk_size": 512, "chunk_overlap": 64},
                        "embedding": {"model_id": "embed-1"},
                        "retrieval": {"method": "vector", "number_of_chunks": 5},
                        "generation": {"model_id": "gen-1"},
                    },
                }
            ),
            encoding="utf-8",
        )
        html_artifact = mock.MagicMock(path=str(tmp_path / "leaderboard_ac.html"))

        leaderboard_evaluation.python_func(
            rag_patterns=str(rag_patterns_dir),
            html_artifact=html_artifact,
            optimization_metric="answer_correctness",
        )

        html_text = Path(html_artifact.path).read_text(encoding="utf-8")
        assert "mean answer correctness" in html_text
        assert "answer correctness" in html_text.lower()

    def test_skips_subdirs_without_pattern_json(self, tmp_path):
        """Subdirectories without pattern.json are ignored when building the leaderboard."""
        rag_patterns_dir = tmp_path / "patterns"
        valid_dir = rag_patterns_dir / "pattern_valid"
        valid_dir.mkdir(parents=True)
        (valid_dir / "pattern.json").write_text(
            json.dumps(
                {
                    "name": "pattern_valid",
                    "final_score": 0.9,
                    "scores": {"faithfulness": {"mean": 0.9}},
                    "settings": {
                        "chunking": {"method": "recursive", "chunk_size": 512, "chunk_overlap": 64},
                        "embedding": {"model_id": "embed-1"},
                        "retrieval": {"method": "vector", "number_of_chunks": 5},
                        "generation": {"model_id": "gen-1"},
                    },
                }
            ),
            encoding="utf-8",
        )
        (rag_patterns_dir / "empty_dir").mkdir()
        (rag_patterns_dir / "not_a_pattern.txt").write_text("skip me", encoding="utf-8")

        html_artifact = mock.MagicMock(path=str(tmp_path / "leaderboard_skip.html"))
        leaderboard_evaluation.python_func(
            rag_patterns=str(rag_patterns_dir),
            html_artifact=html_artifact,
            optimization_metric="faithfulness",
        )

        html_text = Path(html_artifact.path).read_text(encoding="utf-8")
        assert "pattern_valid" in html_text
        assert "1 pattern(s)" in html_text

    def test_notebook_style_call_without_component_status(self, tmp_path):
        """Direct python_func calls without component_status work (notebook usage)."""
        rag_patterns_dir = tmp_path / "patterns"
        rag_patterns_dir.mkdir()

        # Create a minimal valid pattern
        pattern_dir = rag_patterns_dir / "test_pattern"
        pattern_dir.mkdir()
        (pattern_dir / "pattern.json").write_text(
            json.dumps(
                {
                    "name": "test_pattern",
                    "final_score": 0.85,
                    "scores": {"faithfulness": {"mean": 0.85}},
                    "settings": {
                        "chunking": {"method": "recursive"},
                        "embedding": {"model_id": "embed-model"},
                        "retrieval": {"method": "vector"},
                        "generation": {"model_id": "gen-model"},
                    },
                }
            ),
            encoding="utf-8",
        )

        html_artifact = mock.MagicMock(path=str(tmp_path / "leaderboard.html"))

        leaderboard_evaluation.python_func(
            rag_patterns=str(rag_patterns_dir),
            html_artifact=html_artifact,
            component_status=None,
        )

        # Verify HTML output created
        assert Path(html_artifact.path).exists()
        html_text = Path(html_artifact.path).read_text(encoding="utf-8")
        assert "test_pattern" in html_text

        # Verify no component_status.json created
        assert not list(tmp_path.rglob("component_status.json"))
