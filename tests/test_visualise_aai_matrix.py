"""Tests for the FastAAI Python heatmap helper."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PATH = Path(sys.executable)
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
    colour_palette: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the Python heatmap helper with explicit heatmap thresholds."""
    command = [
        str(PYTHON_PATH),
        str(SCRIPT_PATH),
        "--lower-threshold",
        str(lower_threshold),
        "--upper-threshold",
        str(upper_threshold),
    ]
    if colour_palette is not None:
        command.extend(["--colour-palette", colour_palette])
    command.append(str(matrix_path))
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=TEST_ENV,
    )


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
        """Ensure the clustered SVG contains matrix raster content and legend ticks."""
        clustered_svg_path = FIGURES_DIR / "matrix_20" / "FastAAI_matrix_heatmap.svg"
        svg_text = clustered_svg_path.read_text(encoding="utf-8")
        self.assertIn("<image", svg_text)
        self.assertIn(">40<", svg_text)
        self.assertIn(">70<", svg_text)
        self.assertIn(">100<", svg_text)
        self.assertGreater(svg_text.count("<path"), 5)

    def test_clustered_svg_omits_legend_title_and_matrix_filename(self) -> None:
        """Do not render the old legend title or the matrix filename as figure text."""
        clustered_svg_path = FIGURES_DIR / "matrix_20" / "FastAAI_matrix_heatmap.svg"
        svg_text = clustered_svg_path.read_text(encoding="utf-8")
        self.assertNotIn("Raw value", svg_text)
        self.assertNotIn("FastAAI_matrix.txt", svg_text)

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

    def test_visualiser_accepts_custom_colour_palette(self) -> None:
        """Apply a non-default Matplotlib palette through the CLI and still render outputs."""
        output_dir = FIGURES_DIR / "matrix_20_palette"
        output_dir.mkdir(parents=True, exist_ok=True)
        for output_name in ("FastAAI_matrix.txt", *EXPECTED_OUTPUTS):
            output_path = output_dir / output_name
            if output_path.exists():
                output_path.unlink()
        matrix_path = output_dir / "FastAAI_matrix.txt"
        shutil.copy2(MATRIX_20_PATH, matrix_path)

        result = run_visualiser_with_thresholds(
            matrix_path,
            40,
            100,
            colour_palette="viridis",
        )

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

    def test_validate_thresholds_accepts_full_percent_range(self) -> None:
        """Allow the full percentage identity range up to 100."""
        VISUALISER_MODULE.validate_thresholds(40, 100)

    def test_validate_thresholds_rejects_values_above_100(self) -> None:
        """Reject colour thresholds outside the percentage identity range."""
        with self.assertRaises(SystemExit) as context:
            VISUALISER_MODULE.validate_thresholds(40, 101)

        self.assertIn("within [0,100]", str(context.exception))

    def test_validate_colour_palette_accepts_known_name(self) -> None:
        """Accept a valid Matplotlib palette name."""
        VISUALISER_MODULE.validate_colour_palette("Blues")

    def test_validate_colour_palette_rejects_unknown_name(self) -> None:
        """Reject a palette name that Matplotlib does not provide."""
        with self.assertRaises(SystemExit) as context:
            VISUALISER_MODULE.validate_colour_palette("not_a_palette")

        self.assertIn("Unknown Matplotlib colour palette", str(context.exception))

    def test_should_render_sample_labels_respects_hard_cap(self) -> None:
        """Draw sample labels only when the matrix has 20 or fewer samples."""
        self.assertTrue(VISUALISER_MODULE.should_render_sample_labels(20))
        self.assertFalse(VISUALISER_MODULE.should_render_sample_labels(21))
        self.assertFalse(VISUALISER_MODULE.should_render_sample_labels(100))

    def test_build_axis_labels_respects_hard_cap(self) -> None:
        """Return all labels at or below the cap and none above it."""
        labels_20 = [f"sample_{index}" for index in range(20)]
        labels_21 = [f"sample_{index}" for index in range(21)]

        self.assertEqual(
            VISUALISER_MODULE.build_axis_labels(labels_20),
            labels_20,
        )
        self.assertEqual(
            VISUALISER_MODULE.build_axis_labels(labels_21),
            [""] * 21,
        )

    def test_get_heatmap_extent_uses_half_cell_edges(self) -> None:
        """Return image bounds that keep integer indices at cell centres."""
        self.assertEqual(
            VISUALISER_MODULE.get_heatmap_extent(20),
            (-0.5, 19.5, 19.5, -0.5),
        )

    def test_build_colormap_uses_requested_palette(self) -> None:
        """Use the requested palette for the shared heatmap colour scale."""
        cmap = VISUALISER_MODULE.build_colormap("Blues")

        self.assertEqual(cmap.name, "Blues")

    def test_parse_args_defaults_colour_palette_to_blues(self) -> None:
        """Default the CLI heatmap palette to Blues."""
        with mock.patch("sys.argv", [str(SCRIPT_PATH), str(MATRIX_20_PATH)]):
            args = VISUALISER_MODULE.parse_args()

        self.assertEqual(args.colour_palette, "Blues")

    def test_build_distance_condensed_uses_100_minus_identity(self) -> None:
        """Convert percent similarity values to percent distances from 100."""
        matrix_values = np.array(
            [
                [100.0, 68.75, 40.0],
                [68.75, 100.0, 55.5],
                [40.0, 55.5, 100.0],
            ]
        )

        condensed = VISUALISER_MODULE.build_distance_condensed(matrix_values)

        self.assertTrue(
            np.allclose(
                condensed,
                np.array([31.25, 60.0, 44.5]),
            )
        )

    def test_clustered_render_uses_complete_linkage_explicitly(self) -> None:
        """Build the clustered ordering with explicit complete linkage."""
        genome_names = ["A", "B", "C"]
        matrix_values = np.array(
            [
                [100.0, 80.0, 45.0],
                [80.0, 100.0, 50.0],
                [45.0, 50.0, 100.0],
            ]
        )
        with tempfile.TemporaryDirectory() as tempdir:
            output_path = Path(tempdir) / "clustered.svg"
            with mock.patch.object(
                VISUALISER_MODULE,
                "linkage",
                wraps=VISUALISER_MODULE.linkage,
            ) as linkage_mock:
                VISUALISER_MODULE.render_clustered_figure(
                    genome_names,
                    matrix_values,
                    output_path,
                    "clustered.svg",
                    40.0,
                    100.0,
                    "Blues",
                )

        self.assertEqual(linkage_mock.call_args.kwargs["method"], "complete")

    def test_derive_top_dendrogram_height_scales_with_matrix_size(self) -> None:
        """Use a lower capped clustered top dendrogram height."""
        self.assertEqual(VISUALISER_MODULE.derive_top_dendrogram_height(0.3), 0.045)
        self.assertAlmostEqual(
            VISUALISER_MODULE.derive_top_dendrogram_height(0.6),
            0.054,
        )
        self.assertEqual(VISUALISER_MODULE.derive_top_dendrogram_height(1.0), 0.07)

    def test_derive_clustered_base_dimensions_use_taller_top_band_content(self) -> None:
        """Size the top band from whichever is taller: legend or dendrogram."""
        base_dimensions = VISUALISER_MODULE.derive_clustered_base_dimensions(100)

        self.assertEqual(
            base_dimensions["top_band_height_in"],
            max(
                base_dimensions["legend_height_in"],
                base_dimensions["top_dendrogram_height_in"],
            ),
        )
        self.assertEqual(
            base_dimensions["left_column_width_in"],
            max(
                base_dimensions["legend_width_in"],
                base_dimensions["left_dendrogram_width_in"],
            ),
        )

    def test_clustered_base_dimensions_cap_legend_height_below_matrix_height(self) -> None:
        """Keep the clustered legend far shorter than the matrix side."""
        base_dimensions = VISUALISER_MODULE.derive_clustered_base_dimensions(100)

        self.assertLess(
            base_dimensions["legend_height_in"],
            base_dimensions["matrix_side_in"],
        )

    def test_prune_label_intervals_does_not_drop_clipped_labels_without_bounds(self) -> None:
        """Do not hide intervals purely because they would clip before resizing."""
        keep_flags = VISUALISER_MODULE.prune_label_intervals(
            [(-2.0, 5.0), (6.0, 9.0), (9.5, 12.0)],
        )
        self.assertEqual(keep_flags, [True, True, True])

    def test_prune_label_intervals_hides_overlapping_labels(self) -> None:
        """Drop labels that overlap the last kept label."""
        keep_flags = VISUALISER_MODULE.prune_label_intervals(
            [(0.0, 3.0), (2.5, 4.0), (4.1, 5.0)],
        )
        self.assertEqual(keep_flags, [True, False, True])

    def test_prune_label_intervals_preserves_cleanly_fitting_labels(self) -> None:
        """Keep labels that fit inside bounds without overlap."""
        keep_flags = VISUALISER_MODULE.prune_label_intervals(
            [(0.0, 2.0), (2.1, 3.0), (3.1, 5.0)],
        )
        self.assertEqual(keep_flags, [True, True, True])

    def test_measure_axis_label_overhangs_detects_long_bottom_labels(self) -> None:
        """Measure extra bottom space needed for long x labels."""
        figure, axis = plt.subplots(figsize=(3.0, 3.0))
        try:
            axis.set_position([0.15, 0.12, 0.45, 0.45])
            axis.set_xticks([0, 1])
            axis.set_yticks([0, 1])
            axis.set_xticklabels(
                ["very_long_bottom_label_a", "very_long_bottom_label_b"],
                rotation=90,
                fontsize=8,
            )
            axis.set_yticklabels(["a", "b"], fontsize=8)
            axis.yaxis.tick_right()
            figure.canvas.draw()
            right_overhang_in, bottom_overhang_in = (
                VISUALISER_MODULE.measure_axis_label_overhangs(figure, axis)
            )

            self.assertEqual(right_overhang_in, 0.0)
            self.assertGreater(bottom_overhang_in, 0.0)
        finally:
            plt.close(figure)

    def test_measure_axis_label_overhangs_detects_long_right_labels(self) -> None:
        """Measure extra right space needed for long y labels."""
        figure, axis = plt.subplots(figsize=(3.0, 3.0))
        try:
            axis.set_position([0.15, 0.2, 0.45, 0.45])
            axis.set_xticks([0, 1])
            axis.set_yticks([0, 1])
            axis.set_xticklabels(["a", "b"], rotation=90, fontsize=8)
            axis.set_yticklabels(
                ["very_long_right_label_a", "very_long_right_label_b"],
                fontsize=8,
            )
            axis.yaxis.tick_right()
            axis.tick_params(
                axis="y",
                labelleft=False,
                left=False,
                labelright=True,
                right=False,
                pad=2,
            )
            figure.canvas.draw()
            right_overhang_in, bottom_overhang_in = (
                VISUALISER_MODULE.measure_axis_label_overhangs(figure, axis)
            )

            self.assertGreater(right_overhang_in, 0.0)
            self.assertEqual(bottom_overhang_in, 0.0)
        finally:
            plt.close(figure)

    def test_expand_figure_for_labels_grows_for_long_bottom_and_right_labels(self) -> None:
        """Expand clustered layout to fit measured bottom and right label overhang."""
        base_dimensions = VISUALISER_MODULE.derive_clustered_base_dimensions(20)
        layout = VISUALISER_MODULE.derive_clustered_layout(
            base_dimensions,
            right_pad_in=base_dimensions["min_right_pad_in"],
            bottom_pad_in=base_dimensions["min_bottom_pad_in"],
        )
        figure, axis = plt.subplots(
            figsize=(layout["figure_width_in"], layout["figure_height_in"])
        )
        try:
            axis.set_position(
                [
                    layout["matrix_left"],
                    layout["matrix_bottom"],
                    layout["matrix_width"],
                    layout["matrix_height"],
                ]
            )
            axis.set_xticks([0, 1])
            axis.set_yticks([0, 1])
            axis.set_xticklabels(
                ["very_long_bottom_label_a", "very_long_bottom_label_b"],
                rotation=90,
                fontsize=8,
            )
            axis.set_yticklabels(
                ["very_long_right_label_a", "very_long_right_label_b"],
                fontsize=8,
            )
            axis.yaxis.tick_right()
            axis.tick_params(
                axis="y",
                labelleft=False,
                left=False,
                labelright=True,
                right=False,
                pad=2,
            )
            figure.canvas.draw()
            expanded_layout, right_overhang_in, bottom_overhang_in = (
                VISUALISER_MODULE.expand_figure_for_labels(
                    figure,
                    axis,
                    base_dimensions,
                    0.08,
                )
            )

            self.assertGreater(right_overhang_in, 0.0)
            self.assertGreater(bottom_overhang_in, 0.0)
            self.assertGreater(
                expanded_layout["figure_width_in"],
                layout["figure_width_in"],
            )
            self.assertGreater(
                expanded_layout["figure_height_in"],
                layout["figure_height_in"],
            )
        finally:
            plt.close(figure)

    def test_expand_figure_for_labels_does_not_shrink_fitted_layout(self) -> None:
        """Keep the larger figure once labels already fit inside the canvas."""
        base_dimensions = VISUALISER_MODULE.derive_clustered_base_dimensions(20)
        layout = VISUALISER_MODULE.derive_clustered_layout(
            base_dimensions,
            right_pad_in=base_dimensions["min_right_pad_in"] + 1.5,
            bottom_pad_in=base_dimensions["min_bottom_pad_in"] + 1.5,
        )
        figure, axis = plt.subplots(
            figsize=(layout["figure_width_in"], layout["figure_height_in"])
        )
        try:
            axis.set_position(
                [
                    layout["matrix_left"],
                    layout["matrix_bottom"],
                    layout["matrix_width"],
                    layout["matrix_height"],
                ]
            )
            axis.set_xticks([0, 1])
            axis.set_yticks([0, 1])
            axis.set_xticklabels(["a", "b"], rotation=90, fontsize=8)
            axis.set_yticklabels(["c", "d"], fontsize=8)
            axis.yaxis.tick_right()
            axis.tick_params(
                axis="y",
                labelleft=False,
                left=False,
                labelright=True,
                right=False,
                pad=2,
            )
            figure.canvas.draw()
            expanded_layout, right_overhang_in, bottom_overhang_in = (
                VISUALISER_MODULE.expand_figure_for_labels(
                    figure,
                    axis,
                    base_dimensions,
                    0.08,
                )
            )

            self.assertEqual(right_overhang_in, 0.0)
            self.assertEqual(bottom_overhang_in, 0.0)
            self.assertEqual(expanded_layout["figure_width_in"], layout["figure_width_in"])
            self.assertEqual(
                expanded_layout["figure_height_in"],
                layout["figure_height_in"],
            )
        finally:
            plt.close(figure)

    def test_derive_clustered_layout_expands_for_extra_label_space(self) -> None:
        """Grow figure dimensions when measured label overhang increases."""
        base_dimensions = VISUALISER_MODULE.derive_clustered_base_dimensions(100)
        base_layout = VISUALISER_MODULE.derive_clustered_layout(
            base_dimensions,
            right_pad_in=base_dimensions["min_right_pad_in"],
            bottom_pad_in=base_dimensions["min_bottom_pad_in"],
        )
        expanded_layout = VISUALISER_MODULE.derive_clustered_layout(
            base_dimensions,
            right_pad_in=base_dimensions["min_right_pad_in"] + 0.5,
            bottom_pad_in=base_dimensions["min_bottom_pad_in"] + 0.75,
        )

        self.assertGreater(
            expanded_layout["figure_width_in"],
            base_layout["figure_width_in"],
        )
        self.assertGreater(
            expanded_layout["figure_height_in"],
            base_layout["figure_height_in"],
        )

    def test_clustered_layout_places_legend_in_top_left_corner(self) -> None:
        """Place the legend above the left dendrogram rather than beside the matrix body."""
        base_dimensions = VISUALISER_MODULE.derive_clustered_base_dimensions(100)
        layout = VISUALISER_MODULE.derive_clustered_layout(
            base_dimensions,
            right_pad_in=base_dimensions["min_right_pad_in"],
            bottom_pad_in=base_dimensions["min_bottom_pad_in"],
        )

        self.assertLess(layout["legend_left"], layout["top_axis_left"])
        self.assertGreater(layout["legend_bottom"], layout["left_axis_bottom"])
        self.assertLess(layout["legend_height"], layout["left_axis_height"])

    def test_clustered_layout_keeps_legend_within_top_band(self) -> None:
        """Keep the top-left legend inside the clustered top content band."""
        base_dimensions = VISUALISER_MODULE.derive_clustered_base_dimensions(100)
        layout = VISUALISER_MODULE.derive_clustered_layout(
            base_dimensions,
            right_pad_in=base_dimensions["min_right_pad_in"],
            bottom_pad_in=base_dimensions["min_bottom_pad_in"],
        )
        top_band_top = layout["top_axis_bottom"] + (
            base_dimensions["top_band_height_in"] / layout["figure_height_in"]
        )

        self.assertGreaterEqual(layout["legend_bottom"], layout["top_axis_bottom"])
        self.assertLessEqual(layout["legend_bottom"] + layout["legend_height"], top_band_top)

    def test_clustered_layout_matches_top_dendrogram_width_to_matrix(self) -> None:
        """Keep the top dendrogram aligned to the full matrix width."""
        base_dimensions = VISUALISER_MODULE.derive_clustered_base_dimensions(100)
        layout = VISUALISER_MODULE.derive_clustered_layout(
            base_dimensions,
            right_pad_in=base_dimensions["min_right_pad_in"],
            bottom_pad_in=base_dimensions["min_bottom_pad_in"],
        )

        self.assertEqual(layout["top_axis_left"], layout["matrix_left"])
        self.assertEqual(layout["top_axis_width"], layout["matrix_width"])

    def test_prune_clustered_labels_is_no_op_when_all_labels_must_be_kept(self) -> None:
        """Do not prune clustered labels when sample labels are mandatory."""
        figure, axis = plt.subplots(figsize=(3.0, 3.0))
        try:
            axis.set_xticks([0, 1])
            axis.set_yticks([0, 1])
            axis.set_xticklabels(["label_a", "label_b"])
            axis.set_yticklabels(["label_c", "label_d"])
            figure.canvas.draw()
            VISUALISER_MODULE.prune_clustered_labels(
                figure,
                axis,
                show_all_labels=True,
            )

            self.assertEqual(
                [label.get_text() for label in axis.get_xticklabels()],
                ["label_a", "label_b"],
            )
            self.assertEqual(
                [label.get_text() for label in axis.get_yticklabels()],
                ["label_c", "label_d"],
            )
        finally:
            plt.close(figure)

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

    def test_visualiser_rejects_missing_values(self) -> None:
        """Reject missing or NA-like matrix cells instead of imputing them."""
        self.assert_matrix_failure(
            [
                "query_genome\tA\tB",
                "A\t100\tNA",
                "B\t68.75\t100",
            ],
            "Non-numeric FastAAI value",
        )

    def test_visualiser_rejects_values_above_100(self) -> None:
        """Reject matrix values outside the percentage identity range."""
        self.assert_matrix_failure(
            [
                "query_genome\tA\tB",
                "A\t100\t68.75",
                "B\t68.75\t101",
            ],
            "supported raw range [0,100]",
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
