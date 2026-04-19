"""Tests for the FastAAI R heatmap helper."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "visualise_aai_matrix.R"


def write_text(path: Path, content: str) -> None:
    """Write ASCII test content to a file."""
    path.write_text(content, encoding="ascii")


def run_visualiser(matrix_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the R heatmap helper for a given matrix path."""
    return subprocess.run(
        ["Rscript", str(SCRIPT_PATH), str(matrix_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class VisualiseAAIMatrixTests(unittest.TestCase):
    """Verify the R helper accepts raw FastAAI matrices and rejects malformed ones."""

    def test_visualiser_writes_svg_for_valid_matrix(self) -> None:
        """Create an SVG figure for a valid FastAAI-style matrix."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            matrix_path = temp_path / "FastAAI_matrix.txt"
            write_text(
                matrix_path,
                "\n".join(
                    [
                        "query_genome\tA\tB\tC",
                        "A\t95\t68.75\t15",
                        "B\t68.75\t95\t0",
                        "C\t15\t0\t95",
                    ]
                )
                + "\n",
            )

            result = run_visualiser(matrix_path)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            svg_path = temp_path / "FastAAI_matrix_heatmap.svg"
            self.assertTrue(svg_path.is_file())
            self.assertGreater(svg_path.stat().st_size, 0)

    def test_visualiser_rejects_missing_query_genome_header(self) -> None:
        """Reject matrices without the FastAAI header marker."""
        self.assert_matrix_failure(
            [
                "genome\tA\tB",
                "A\t95\t68.75",
                "B\t68.75\t95",
            ],
            "first header cell must be 'query_genome'",
        )

    def test_visualiser_rejects_duplicate_names(self) -> None:
        """Reject duplicate genome names in the header."""
        self.assert_matrix_failure(
            [
                "query_genome\tA\tA",
                "A\t95\t68.75",
                "A\t68.75\t95",
            ],
            "duplicate genome names",
        )

    def test_visualiser_rejects_row_header_mismatch(self) -> None:
        """Reject matrices whose row names do not match the header order."""
        self.assert_matrix_failure(
            [
                "query_genome\tA\tB",
                "A\t95\t68.75",
                "C\t68.75\t95",
            ],
            "does not match header name",
        )

    def test_visualiser_rejects_non_numeric_values(self) -> None:
        """Reject matrices with non-numeric FastAAI cells."""
        self.assert_matrix_failure(
            [
                "query_genome\tA\tB",
                "A\t95\tbad",
                "B\t68.75\t95",
            ],
            "Non-numeric FastAAI value",
        )

    def test_visualiser_rejects_asymmetric_values(self) -> None:
        """Reject matrices that are not symmetric."""
        self.assert_matrix_failure(
            [
                "query_genome\tA\tB",
                "A\t95\t68.75",
                "B\t15\t95",
            ],
            "not symmetric",
        )

    def assert_matrix_failure(self, rows: list[str], expected_error: str) -> None:
        """Write a matrix fixture and assert that the helper fails clearly."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            matrix_path = temp_path / "FastAAI_matrix.txt"
            write_text(matrix_path, "\n".join(rows) + "\n")

            result = run_visualiser(matrix_path)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(expected_error, result.stderr)


if __name__ == "__main__":
    unittest.main()
