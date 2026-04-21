"""Tests for the FastAAI Python heatmap helper."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PATH = Path("/Users/asuq/miniforge3/envs/fastaai-debug/bin/python")
SCRIPT_PATH = REPO_ROOT / "scripts" / "visualise_aai_matrix.py"
ASSET_HELPER_PATH = REPO_ROOT / "tests" / "generate_visualiser_test_assets.py"
REAL_MATRIX_PATH = REPO_ROOT / "tests" / "data" / "matrix.tsv"
MATRIX_20_PATH = REPO_ROOT / "tests" / "data" / "matrix_20.tsv"
MATRIX_100_PATH = REPO_ROOT / "tests" / "data" / "matrix_100.tsv"
FIGURES_DIR = REPO_ROOT / "tests" / "figures"
EXPECTED_OUTPUTS = (
    "FastAAI_matrix_heatmap.svg",
    "FastAAI_matrix_heatmap.png",
    "FastAAI_matrix_heatmap_simple.svg",
    "FastAAI_matrix_heatmap_simple.png",
)
TEST_ENV = {
    **os.environ,
    "MPLCONFIGDIR": "/tmp/fastaai_mplconfig",
}


def load_visualiser_module():
    """Load the visualiser script as a Python module for helper tests."""
    spec = importlib.util.spec_from_file_location("visualise_aai_matrix", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VISUALISER_MODULE = load_visualiser_module()


def write_text(path: Path, content: str) -> None:
    """Write ASCII test content to a file."""
    path.write_text(content, encoding="ascii")


def run_visualiser(matrix_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the Python heatmap helper for a given matrix path."""
    return subprocess.run(
        [str(PYTHON_PATH), str(SCRIPT_PATH), str(matrix_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=TEST_ENV,
    )


def refresh_visualiser_assets() -> subprocess.CompletedProcess[str]:
    """Regenerate stable test matrices and figures under tests/."""
    return subprocess.run(
        [str(PYTHON_PATH), str(ASSET_HELPER_PATH)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=TEST_ENV,
    )


def run_visualiser_with_thresholds(
    matrix_path: Path,
    lower_threshold: float,
    upper_threshold: float,
) -> subprocess.CompletedProcess[str]:
    """Run the Python heatmap helper with explicit heatmap thresholds."""
    return subprocess.run(
        [
            str(PYTHON_PATH),
            str(SCRIPT_PATH),
            "--lower-threshold",
            str(lower_threshold),
            "--upper-threshold",
            str(upper_threshold),
            str(matrix_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=TEST_ENV,
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


class VisualiseAAIMatrixTests(unittest.TestCase):
    """Verify the Python helper accepts raw FastAAI matrices and rejects malformed ones."""

    @classmethod
    def setUpClass(cls) -> None:
        """Regenerate the stable visualiser test assets once for the test class."""
        result = refresh_visualiser_assets()
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout or "Asset refresh failed")

    def test_visualiser_writes_both_svgs_for_20_sample_real_submatrix(self) -> None:
        """Create SVG and PNG figures for the committed 20-sample real submatrix."""
        self.assert_render_outputs_exist("matrix_20", MATRIX_20_PATH)

    def test_visualiser_writes_both_svgs_for_100_sample_real_submatrix(self) -> None:
        """Create SVG and PNG figures for the committed 100-sample real submatrix."""
        self.assert_render_outputs_exist("matrix_100", MATRIX_100_PATH)

    def test_visualiser_writes_both_svgs_for_full_real_matrix(self) -> None:
        """Run the actual integration-style render test on the full real matrix fixture."""
        self.assert_render_outputs_exist("matrix_full", REAL_MATRIX_PATH)

    def test_simple_svg_contains_matrix_draw_commands_for_real_submatrix(self) -> None:
        """Emit a real-data simple SVG that is larger than an empty shell and contains draw paths."""
        simple_svg_path = FIGURES_DIR / "matrix_20" / "FastAAI_matrix_heatmap_simple.svg"
        svg_text = simple_svg_path.read_text(encoding="utf-8")
        self.assertGreater(simple_svg_path.stat().st_size, 5000)
        self.assertIn("<image", svg_text)

    def test_clustered_svg_contains_matrix_and_legend_elements(self) -> None:
        """Ensure the clustered SVG contains matrix raster content and legend text."""
        clustered_svg_path = FIGURES_DIR / "matrix_20" / "FastAAI_matrix_heatmap.svg"
        svg_text = clustered_svg_path.read_text(encoding="utf-8")
        self.assertIn("<image", svg_text)
        self.assertIn("Raw value", svg_text)
        self.assertGreater(svg_text.count("<path"), 5)

    def test_visualiser_accepts_custom_heatmap_thresholds(self) -> None:
        """Apply non-default heatmap thresholds through the CLI and still render outputs."""
        output_dir = FIGURES_DIR / "matrix_20_thresholds"
        output_dir.mkdir(parents=True, exist_ok=True)
        for output_name in ("FastAAI_matrix.txt", *EXPECTED_OUTPUTS):
            output_path = output_dir / output_name
            if output_path.exists():
                output_path.unlink()
        matrix_path = output_dir / "FastAAI_matrix.txt"
        shutil.copy2(MATRIX_20_PATH, matrix_path)

        result = run_visualiser_with_thresholds(matrix_path, 35, 80)

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assert_render_files_non_empty(output_dir)

    def test_visualiser_rejects_inverted_thresholds(self) -> None:
        """Reject heatmap thresholds when the lower threshold is not smaller than the upper."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            matrix_path = temp_path / "FastAAI_matrix.txt"
            matrix_path.write_text(MATRIX_20_PATH.read_text(encoding="utf-8"), encoding="utf-8")

            result = run_visualiser_with_thresholds(matrix_path, 90, 30)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Lower threshold must be smaller than upper threshold", result.stderr)

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

    def test_get_heatmap_extent_uses_half_cell_edges(self) -> None:
        """Return image bounds that keep integer indices at cell centres."""
        self.assertEqual(
            VISUALISER_MODULE.get_heatmap_extent(20),
            (-0.5, 19.5, 19.5, -0.5),
        )

    def test_build_dendrogram_segments_maps_scipy_positions_to_integer_centres(self) -> None:
        """Map SciPy dendrogram leaf coordinates onto integer cell centres."""
        segments = VISUALISER_MODULE.build_dendrogram_segments(
            icoord=[[5.0, 5.0, 15.0, 15.0]],
            dcoord=[[0.0, 2.0, 2.0, 0.0]],
            orientation="top",
        )
        self.assertEqual(segments[0][:, 0].tolist(), [0.0, 0.0, 1.0, 1.0])

    def test_apply_dendrogram_limits_uses_heatmap_edge_span(self) -> None:
        """Use the matrix edge span rather than centre span for dendrogram axes."""
        figure, axes = plt.subplots(1, 2)
        try:
            VISUALISER_MODULE.apply_dendrogram_limits(
                axes[0],
                leaf_count=20,
                max_height=7.5,
                orientation="top",
            )
            VISUALISER_MODULE.apply_dendrogram_limits(
                axes[1],
                leaf_count=20,
                max_height=7.5,
                orientation="left",
            )
            self.assertEqual(axes[0].get_xlim(), (-0.5, 19.5))
            self.assertEqual(axes[0].get_ylim(), (0.0, 7.5))
            self.assertEqual(axes[1].get_xlim(), (7.5, 0.0))
            self.assertEqual(axes[1].get_ylim(), (19.5, -0.5))
        finally:
            plt.close(figure)

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

    def assert_render_outputs_exist(self, figure_dir_name: str, source_matrix_path: Path) -> None:
        """Verify a stable figure directory exists for a committed matrix fixture."""
        self.assertTrue(source_matrix_path.is_file())
        figure_dir = FIGURES_DIR / figure_dir_name
        self.assertTrue(figure_dir.is_dir())
        self.assertTrue((figure_dir / "FastAAI_matrix.txt").is_file())
        self.assert_render_files_non_empty(figure_dir)

    def assert_render_files_non_empty(self, output_dir: Path) -> None:
        """Verify the expected figure outputs exist and are non-empty."""
        for output_name in EXPECTED_OUTPUTS:
            output_path = output_dir / output_name
            self.assertTrue(output_path.is_file(), msg=f"Missing output: {output_path}")
            self.assertGreater(output_path.stat().st_size, 0, msg=f"Empty output: {output_path}")


if __name__ == "__main__":
    unittest.main()
