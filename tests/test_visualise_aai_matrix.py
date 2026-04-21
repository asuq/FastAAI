"""Tests for the FastAAI R heatmap helper."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from math import floor
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "visualise_aai_matrix.R"
REAL_MATRIX_PATH = REPO_ROOT / "tests" / "data" / "matrix.tsv"


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


def expected_label_indices(genome_count: int) -> list[int]:
    """Return the expected zero-based label positions for a matrix size."""
    if genome_count <= 60:
        stride = 1
    elif genome_count <= 150:
        stride = 5
    elif genome_count <= 400:
        stride = 10
    else:
        stride = 25

    keep = list(range(0, genome_count, stride))
    if keep[-1] != genome_count - 1:
        keep.append(genome_count - 1)
    return keep


def load_real_matrix_rows() -> list[list[str]]:
    """Load the real FastAAI matrix fixture as tab-separated rows."""
    return [
        line.rstrip("\n").split("\t")
        for line in REAL_MATRIX_PATH.read_text(encoding="utf-8").splitlines()
        if line
    ]


def evenly_spaced_indices(total_size: int, subset_size: int) -> list[int]:
    """Return deterministic evenly spaced zero-based indices across a range."""
    if subset_size <= 0 or subset_size > total_size:
        raise ValueError("subset_size must be between 1 and total_size")
    if subset_size == 1:
        return [0]
    return sorted(
        {
            min(total_size - 1, floor(i * (total_size - 1) / (subset_size - 1)))
            for i in range(subset_size)
        }
    )


def write_submatrix(path: Path, rows: list[list[str]], data_indices: list[int]) -> None:
    """Write a square FastAAI submatrix using the selected zero-based data indices."""
    header = rows[0]
    selected_header = [header[0]] + [header[index + 1] for index in data_indices]
    selected_rows = ["\t".join(selected_header)]
    for row_index in data_indices:
        row = rows[row_index + 1]
        selected_row = [row[0]] + [row[index + 1] for index in data_indices]
        selected_rows.append("\t".join(selected_row))
    path.write_text("\n".join(selected_rows) + "\n", encoding="utf-8")


class VisualiseAAIMatrixTests(unittest.TestCase):
    """Verify the R helper accepts raw FastAAI matrices and rejects malformed ones."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the real matrix fixture once for derived valid-render tests."""
        cls.real_matrix_rows = load_real_matrix_rows()
        cls.real_matrix_size = len(cls.real_matrix_rows) - 1

    def test_visualiser_writes_both_svgs_for_20_sample_real_submatrix(self) -> None:
        """Create both SVG figures for an evenly spaced 20-sample real submatrix."""
        self.assert_real_submatrix_renders(20)

    def test_visualiser_writes_both_svgs_for_100_sample_real_submatrix(self) -> None:
        """Create both SVG figures for an evenly spaced 100-sample real submatrix."""
        self.assert_real_submatrix_renders(100)

    def test_visualiser_writes_both_svgs_for_full_real_matrix(self) -> None:
        """Run the actual integration-style render test on the full real matrix fixture."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            matrix_path = temp_path / "FastAAI_matrix.txt"
            matrix_path.write_text(REAL_MATRIX_PATH.read_text(encoding="utf-8"), encoding="utf-8")

            result = run_visualiser(matrix_path)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            clustered_svg_path = temp_path / "FastAAI_matrix_heatmap.svg"
            simple_svg_path = temp_path / "FastAAI_matrix_heatmap_simple.svg"
            self.assertTrue(clustered_svg_path.is_file())
            self.assertTrue(simple_svg_path.is_file())
            self.assertGreater(clustered_svg_path.stat().st_size, 0)
            self.assertGreater(simple_svg_path.stat().st_size, 0)

    def test_simple_svg_contains_matrix_draw_commands_for_real_submatrix(self) -> None:
        """Emit a real-data simple SVG that is larger than an empty shell and contains draw paths."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            matrix_path = temp_path / "FastAAI_matrix.txt"
            write_submatrix(
                matrix_path,
                self.real_matrix_rows,
                evenly_spaced_indices(self.real_matrix_size, 20),
            )

            result = run_visualiser(matrix_path)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            simple_svg_path = temp_path / "FastAAI_matrix_heatmap_simple.svg"
            svg_text = simple_svg_path.read_text(encoding="utf-8")
            self.assertGreater(simple_svg_path.stat().st_size, 5000)
            self.assertGreater(svg_text.count("<path"), 5)

    def test_expected_label_indices_follow_thinning_policy(self) -> None:
        """Keep every nth label with the first and last labels always preserved."""
        expected = {
            20: list(range(20)),
            100: list(range(0, 100, 5)),
            250: list(range(0, 250, 10)),
            1000: list(range(0, 1000, 25)),
        }
        expected[100].append(99)
        expected[250].append(249)
        expected[1000].append(999)

        for genome_count, keep in expected.items():
            with self.subTest(genome_count=genome_count):
                self.assertEqual(expected_label_indices(genome_count), keep)

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

    def assert_real_submatrix_renders(self, subset_size: int) -> None:
        """Render a deterministic real-data submatrix and verify both SVG outputs."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            matrix_path = temp_path / "FastAAI_matrix.txt"
            write_submatrix(
                matrix_path,
                self.real_matrix_rows,
                evenly_spaced_indices(self.real_matrix_size, subset_size),
            )

            result = run_visualiser(matrix_path)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            clustered_svg_path = temp_path / "FastAAI_matrix_heatmap.svg"
            simple_svg_path = temp_path / "FastAAI_matrix_heatmap_simple.svg"
            self.assertTrue(clustered_svg_path.is_file())
            self.assertTrue(simple_svg_path.is_file())
            self.assertGreater(clustered_svg_path.stat().st_size, 0)
            self.assertGreater(simple_svg_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
