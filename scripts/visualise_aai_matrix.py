#!/usr/bin/env python3
"""Visualise a raw FastAAI matrix as clustered and simple heatmaps."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.artist import Artist
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


EXPECTED_HEADER = "query_genome"
MAX_LABELLED_SAMPLES = 20
OUTPUT_FILENAMES = (
    "FastAAI_matrix_heatmap.svg",
    "FastAAI_matrix_heatmap.png",
    "FastAAI_matrix_heatmap_simple.svg",
    "FastAAI_matrix_heatmap_simple.png",
)
SVG_FONT_TYPE = "none"
SYMMETRY_TOLERANCE = 1e-8


def die(message: str) -> "NoReturn":
    """Exit with a concise error message."""
    raise SystemExit(message)


def parse_args() -> argparse.Namespace:
    """Parse the visualiser command line."""
    parser = argparse.ArgumentParser(
        description="Visualise a raw FastAAI matrix as paired SVG and PNG heatmaps.",
    )
    parser.add_argument(
        "matrix_path",
        nargs="?",
        default="FastAAI_matrix.txt",
        help="Path to the raw FastAAI matrix file.",
    )
    parser.add_argument(
        "--lower-threshold",
        type=float,
        default=40.0,
        help="Lower heatmap threshold for colour clipping.",
    )
    parser.add_argument(
        "--upper-threshold",
        type=float,
        default=100.0,
        help="Upper heatmap threshold for colour clipping.",
    )
    parser.add_argument(
        "--colour-palette",
        "--color-palette",
        dest="colour_palette",
        default="magma",
        help="Matplotlib colour palette name for the heatmaps.",
    )
    args = parser.parse_args()
    validate_thresholds(args.lower_threshold, args.upper_threshold)
    validate_colour_palette(args.colour_palette)
    return args


def validate_thresholds(lower_threshold: float, upper_threshold: float) -> None:
    """Validate the user-supplied threshold range."""
    if not 0 <= lower_threshold <= 100:
        die(f"Lower threshold must be within [0,100], got: {lower_threshold}")
    if not 0 <= upper_threshold <= 100:
        die(f"Upper threshold must be within [0,100], got: {upper_threshold}")
    if lower_threshold >= upper_threshold:
        die(
            "Lower threshold must be smaller than upper threshold, "
            f"got: {lower_threshold} >= {upper_threshold}"
        )


def validate_colour_palette(colour_palette: str) -> None:
    """Validate the requested Matplotlib colour palette name."""
    if colour_palette not in plt.colormaps():
        die(
            "Unknown Matplotlib colour palette: "
            f"'{colour_palette}'. Choose one of: {', '.join(sorted(plt.colormaps()))}"
        )


def read_matrix_file(matrix_path: Path) -> tuple[list[str], np.ndarray]:
    """Read and validate a raw FastAAI matrix file."""
    if not matrix_path.is_file():
        die(f"FastAAI matrix file not found: {matrix_path}")

    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.reader(handle, delimiter="\t") if row]

    if not rows:
        die("FastAAI matrix is empty.")

    header = rows[0]
    if len(header) < 2:
        die("FastAAI matrix header must contain 'query_genome' plus at least one genome name.")
    if header[0] != EXPECTED_HEADER:
        die(
            "FastAAI matrix first header cell must be "
            f"'{EXPECTED_HEADER}', got: '{header[0]}'"
        )

    genome_names = header[1:]
    if len(genome_names) < 2:
        die("FastAAI matrix must contain at least two genomes.")

    duplicates = sorted({name for name in genome_names if genome_names.count(name) > 1})
    if duplicates:
        die(
            "FastAAI matrix header contains duplicate genome names: "
            + ", ".join(duplicates)
        )

    matrix_rows = rows[1:]
    if len(matrix_rows) != len(genome_names):
        die(
            "FastAAI matrix must be square: "
            f"expected {len(genome_names)} data rows, found {len(matrix_rows)}."
        )

    matrix_values = np.empty((len(genome_names), len(genome_names)), dtype=float)
    expected_width = len(genome_names) + 1

    for row_index, row_fields in enumerate(matrix_rows):
        if len(row_fields) != expected_width:
            die(
                f"FastAAI matrix row {row_index + 1} has {len(row_fields)} columns, "
                f"expected {expected_width}."
            )

        row_name = row_fields[0]
        expected_name = genome_names[row_index]
        if row_name != expected_name:
            die(
                f"FastAAI matrix row {row_index + 1} name '{row_name}' does not match "
                f"header name '{expected_name}' at the same position."
            )

        numeric_values: list[float] = []
        for column_index, raw_value in enumerate(row_fields[1:]):
            try:
                value = float(raw_value)
            except ValueError:
                die(
                    "Non-numeric FastAAI value at row "
                    f"'{row_name}', column '{genome_names[column_index]}': '{raw_value}'"
                )
            if not np.isfinite(value):
                die(
                    "Non-numeric FastAAI value at row "
                    f"'{row_name}', column '{genome_names[column_index]}': '{raw_value}'"
                )
            if value < 0 or value > 100:
                die(
                    "FastAAI value out of the supported raw range [0,100] at row "
                    f"'{row_name}', column '{genome_names[column_index]}': {raw_value}"
                )
            numeric_values.append(value)

        matrix_values[row_index, :] = numeric_values

    if not np.allclose(matrix_values, matrix_values.T, atol=SYMMETRY_TOLERANCE, rtol=0):
        mismatch = np.argwhere(
            np.abs(matrix_values - matrix_values.T) > SYMMETRY_TOLERANCE
        )[0]
        row_index, column_index = int(mismatch[0]), int(mismatch[1])
        die(
            "FastAAI matrix is not symmetric: "
            f"{genome_names[row_index]} vs {genome_names[column_index]} "
            f"({matrix_values[row_index, column_index]:g} != "
            f"{matrix_values[column_index, row_index]:g})"
        )

    return genome_names, matrix_values


def build_distance_condensed(matrix_values: np.ndarray) -> np.ndarray:
    """Convert percent FastAAI similarity values into condensed distances."""
    distance_values = 100.0 - matrix_values
    distance_values[(distance_values < 0) & (distance_values > -SYMMETRY_TOLERANCE)] = 0
    if np.any(distance_values < 0):
        die("Derived negative distances from the FastAAI matrix.")
    np.fill_diagonal(distance_values, 0)
    return squareform(distance_values, checks=False)


def should_render_sample_labels(genome_count: int) -> bool:
    """Return whether sample labels should be drawn for this matrix size."""
    return genome_count <= MAX_LABELLED_SAMPLES


def build_axis_labels(genome_names: list[str]) -> list[str]:
    """Return sample labels or suppress them entirely above the hard cap."""
    genome_count = len(genome_names)
    if should_render_sample_labels(genome_count):
        return genome_names
    return [""] * genome_count


def derive_label_size(genome_count: int) -> float:
    """Return a matrix label size for the current matrix density."""
    if genome_count <= 60:
        return 6.0
    if genome_count <= 150:
        return 4.5
    if genome_count <= 400:
        return 3.5
    return 2.6


def derive_device_size(genome_count: int, simple: bool = False) -> float:
    """Choose a square figure size in inches."""
    if simple:
        return max(8.0, min(32.0, 8.0 + genome_count * 0.025))
    return max(10.0, min(40.0, 10.0 + genome_count * 0.03))


def derive_top_dendrogram_height(matrix_size: float) -> float:
    """Return the clustered top dendrogram height as a figure fraction."""
    return max(0.045, min(0.07, matrix_size * 0.09))


def derive_clustered_base_dimensions(genome_count: int) -> dict[str, float]:
    """Return compact baseline clustered layout dimensions in inches."""
    base_square_in = derive_device_size(genome_count, simple=False)
    left_outer_pad_in = max(0.12, min(0.2, base_square_in * 0.015))
    legend_width_in = max(0.7, min(1.1, base_square_in * 0.08))
    left_dendrogram_width_in = max(0.4, min(0.7, base_square_in * 0.05))
    min_right_pad_in = max(0.1, min(0.16, base_square_in * 0.012))
    min_bottom_pad_in = max(0.1, min(0.16, base_square_in * 0.012))
    top_outer_pad_in = max(0.08, min(0.14, base_square_in * 0.01))
    legend_height_in = max(0.95, min(1.5, base_square_in * 0.12))
    left_column_width_in = max(left_dendrogram_width_in, legend_width_in)

    left_content_in = (
        left_outer_pad_in
        + left_column_width_in
    )
    provisional_matrix_side_in = base_square_in - left_content_in - min_right_pad_in
    top_dendrogram_height_in = (
        derive_top_dendrogram_height(provisional_matrix_side_in / base_square_in)
        * base_square_in
    )
    top_band_height_in = max(
        top_dendrogram_height_in,
        legend_height_in,
    )
    top_content_in = top_outer_pad_in + top_band_height_in
    matrix_side_in = min(
        provisional_matrix_side_in,
        base_square_in - top_content_in - min_bottom_pad_in,
    )
    top_dendrogram_height_in = (
        derive_top_dendrogram_height(matrix_side_in / base_square_in)
        * base_square_in
    )
    top_band_height_in = max(
        top_dendrogram_height_in,
        legend_height_in,
    )
    top_content_in = top_outer_pad_in + top_band_height_in

    return {
        "base_square_in": base_square_in,
        "left_outer_pad_in": left_outer_pad_in,
        "legend_width_in": legend_width_in,
        "left_dendrogram_width_in": left_dendrogram_width_in,
        "left_column_width_in": left_column_width_in,
        "min_right_pad_in": min_right_pad_in,
        "min_bottom_pad_in": min_bottom_pad_in,
        "top_outer_pad_in": top_outer_pad_in,
        "legend_height_in": legend_height_in,
        "left_content_in": left_content_in,
        "top_band_height_in": top_band_height_in,
        "top_content_in": top_content_in,
        "matrix_side_in": matrix_side_in,
        "top_dendrogram_height_in": top_dendrogram_height_in,
    }


def build_colormap(colour_palette: str) -> colors.Colormap:
    """Build the shared FastAAI heatmap palette."""
    return plt.get_cmap(colour_palette).copy()


def build_legend_breaks(lower_threshold: float, upper_threshold: float) -> list[float]:
    """Return the compact legend break points."""
    midpoint = round((lower_threshold + upper_threshold) / 2, 2)
    return sorted({lower_threshold, midpoint, upper_threshold})


def get_heatmap_extent(leaf_count: int) -> tuple[float, float, float, float]:
    """Return the explicit image extent for a square heatmap."""
    return (-0.5, leaf_count - 0.5, leaf_count - 0.5, -0.5)


def prune_label_intervals(
    intervals: list[tuple[float, float]],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> list[bool]:
    """Keep intervals that fit optional bounds and do not overlap."""
    keep_flags: list[bool] = []
    last_kept_end: float | None = None
    for start, end in intervals:
        fits_lower_bound = lower_bound is None or start >= lower_bound
        fits_upper_bound = upper_bound is None or end <= upper_bound
        fits_bounds = fits_lower_bound and fits_upper_bound
        overlaps_previous = last_kept_end is not None and start < last_kept_end
        keep_label = fits_bounds and not overlaps_previous
        keep_flags.append(keep_label)
        if keep_label:
            last_kept_end = end
    return keep_flags


def configure_matrix_axis(
    axis: plt.Axes,
    genome_names: list[str],
    label_size: float,
    show_grid: bool,
    row_labels_right: bool = False,
) -> None:
    """Apply shared axis styling for a heatmap matrix."""
    positions = np.arange(len(genome_names))
    axis.set_xticks(positions)
    axis.set_yticks(positions)
    axis.set_xticklabels(build_axis_labels(genome_names), rotation=90, fontsize=label_size)
    axis.set_yticklabels(build_axis_labels(genome_names), fontsize=label_size)
    axis.tick_params(length=0)
    if row_labels_right:
        axis.yaxis.tick_right()
        axis.tick_params(
            axis="y",
            labelleft=False,
            left=False,
            labelright=True,
            right=False,
            pad=2,
        )
        for label in axis.get_yticklabels():
            label.set_horizontalalignment("left")
    else:
        axis.yaxis.tick_left()
        axis.tick_params(
            axis="y",
            labelleft=True,
            left=False,
            labelright=False,
            right=False,
            pad=2,
        )
    if show_grid:
        axis.set_xticks(np.arange(-0.5, len(genome_names), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(genome_names), 1), minor=True)
        axis.grid(which="minor", color="#ffffff66", linewidth=0.35)
        axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#303030")
        spine.set_linewidth(0.8)


def draw_matrix(
    axis: plt.Axes,
    matrix_values: np.ndarray,
    cmap: colors.LinearSegmentedColormap,
    norm: colors.Normalize,
    genome_names: list[str],
    show_grid: bool,
    row_labels_right: bool = False,
) -> None:
    """Render a heatmap matrix panel."""
    heatmap_extent = get_heatmap_extent(len(genome_names))
    axis.imshow(
        matrix_values,
        cmap=cmap,
        norm=norm,
        extent=heatmap_extent,
        interpolation="nearest",
        origin="upper",
        aspect="equal",
    )
    configure_matrix_axis(
        axis,
        genome_names,
        derive_label_size(len(genome_names)),
        show_grid,
        row_labels_right=row_labels_right,
    )


def draw_simple_matrix(
    axis: plt.Axes,
    matrix_values: np.ndarray,
    cmap: colors.Colormap,
    norm: colors.Normalize,
) -> None:
    """Render the simple matrix without labels."""
    heatmap_extent = get_heatmap_extent(matrix_values.shape[0])
    axis.imshow(
        matrix_values,
        cmap=cmap,
        norm=norm,
        extent=heatmap_extent,
        interpolation="nearest",
        origin="upper",
        aspect="equal",
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#303030")
        spine.set_linewidth(0.8)


def style_dendrogram_axis(axis: plt.Axes) -> None:
    """Remove clutter from a dendrogram axis."""
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def build_dendrogram_segments(
    icoord: list[list[float]],
    dcoord: list[list[float]],
    orientation: str,
) -> list[np.ndarray]:
    """Convert SciPy dendrogram coordinates to matrix-centred segments."""
    base_scale = 10.0
    base_offset = 5.0
    segments: list[np.ndarray] = []

    for x_coordinates, y_coordinates in zip(icoord, dcoord, strict=True):
        remapped_positions = [
            ((coordinate - base_offset) / base_scale)
            for coordinate in x_coordinates
        ]
        if orientation == "top":
            segment = np.column_stack([remapped_positions, y_coordinates])
        elif orientation == "left":
            segment = np.column_stack([y_coordinates, remapped_positions])
        else:
            raise ValueError(f"Unsupported dendrogram orientation: {orientation}")
        segments.append(segment)

    return segments


def apply_dendrogram_limits(
    axis: plt.Axes,
    leaf_count: int,
    max_height: float,
    orientation: str,
) -> None:
    """Apply matrix-aligned bounds to a dendrogram axis."""
    left_edge, right_edge, bottom_edge, top_edge = get_heatmap_extent(leaf_count)
    if orientation == "top":
        axis.set_xlim(left_edge, right_edge)
        axis.set_ylim(0.0, max_height)
    elif orientation == "left":
        axis.set_xlim(max_height, 0.0)
        axis.set_ylim(bottom_edge, top_edge)
    else:
        raise ValueError(f"Unsupported dendrogram orientation: {orientation}")

    axis.margins(x=0, y=0)
    axis.set_autoscale_on(False)


def measure_axis_label_overhangs(
    figure: plt.Figure,
    matrix_axis: plt.Axes,
) -> tuple[float, float]:
    """Return right and bottom label overhang in inches."""
    renderer = figure.canvas.get_renderer()
    figure_box = figure.bbox
    right_overhang_px = 0.0
    bottom_overhang_px = 0.0

    for label in matrix_axis.get_yticklabels():
        if not label.get_text():
            continue
        bbox = label.get_window_extent(renderer=renderer)
        right_overhang_px = max(right_overhang_px, bbox.x1 - figure_box.x1)

    for label in matrix_axis.get_xticklabels():
        if not label.get_text():
            continue
        bbox = label.get_window_extent(renderer=renderer)
        bottom_overhang_px = max(bottom_overhang_px, figure_box.y0 - bbox.y0)

    return (
        max(0.0, right_overhang_px) / figure.dpi,
        max(0.0, bottom_overhang_px) / figure.dpi,
    )


def expand_figure_for_labels(
    figure: plt.Figure,
    matrix_axis: plt.Axes,
    base_dimensions: dict[str, float],
    safety_pad_in: float,
) -> tuple[dict[str, float], float, float]:
    """Expand the clustered figure without shrinking an already fitted layout."""
    right_overhang_in, bottom_overhang_in = measure_axis_label_overhangs(
        figure,
        matrix_axis,
    )
    current_width_in, current_height_in = figure.get_size_inches()
    current_right_pad_in = max(
        base_dimensions["min_right_pad_in"],
        current_width_in
        - base_dimensions["left_content_in"]
        - base_dimensions["matrix_side_in"],
    )
    current_bottom_pad_in = max(
        base_dimensions["min_bottom_pad_in"],
        current_height_in
        - base_dimensions["top_content_in"]
        - base_dimensions["matrix_side_in"],
    )
    final_layout = derive_clustered_layout(
        base_dimensions,
        right_pad_in=max(
            current_right_pad_in,
            base_dimensions["min_right_pad_in"] + right_overhang_in + safety_pad_in,
        ),
        bottom_pad_in=max(
            current_bottom_pad_in,
            base_dimensions["min_bottom_pad_in"] + bottom_overhang_in + safety_pad_in,
        ),
    )
    return final_layout, right_overhang_in, bottom_overhang_in


def derive_clustered_layout(
    base_dimensions: dict[str, float],
    right_pad_in: float,
    bottom_pad_in: float,
) -> dict[str, float]:
    """Return final clustered figure size and axes positions."""
    matrix_side_in = base_dimensions["matrix_side_in"]
    left_content_in = base_dimensions["left_content_in"]
    top_content_in = base_dimensions["top_content_in"]
    figure_width_in = left_content_in + matrix_side_in + right_pad_in
    figure_height_in = top_content_in + matrix_side_in + bottom_pad_in

    legend_left_in = base_dimensions["left_outer_pad_in"]
    legend_bottom_in = (
        bottom_pad_in
        + matrix_side_in
        + base_dimensions["top_band_height_in"]
        - base_dimensions["legend_height_in"]
    )
    legend_left = legend_left_in / figure_width_in
    legend_bottom = legend_bottom_in / figure_height_in
    legend_width = base_dimensions["legend_width_in"] / figure_width_in
    legend_height = base_dimensions["legend_height_in"] / figure_height_in

    matrix_left_in = (
        base_dimensions["left_outer_pad_in"]
        + base_dimensions["left_column_width_in"]
    )
    matrix_left = matrix_left_in / figure_width_in
    matrix_bottom = bottom_pad_in / figure_height_in
    matrix_width = matrix_side_in / figure_width_in
    matrix_height = matrix_side_in / figure_height_in

    top_axis_left = matrix_left
    top_axis_bottom = (bottom_pad_in + matrix_side_in) / figure_height_in
    top_axis_width = matrix_width
    top_axis_height = base_dimensions["top_dendrogram_height_in"] / figure_height_in

    left_axis_left = (
        matrix_left_in - base_dimensions["left_dendrogram_width_in"]
    ) / figure_width_in
    left_axis_bottom = matrix_bottom
    left_axis_width = base_dimensions["left_dendrogram_width_in"] / figure_width_in
    left_axis_height = matrix_height

    return {
        "figure_width_in": figure_width_in,
        "figure_height_in": figure_height_in,
        "legend_left": legend_left,
        "legend_bottom": legend_bottom,
        "legend_width": legend_width,
        "legend_height": legend_height,
        "matrix_left": matrix_left,
        "matrix_bottom": matrix_bottom,
        "matrix_width": matrix_width,
        "matrix_height": matrix_height,
        "top_axis_left": top_axis_left,
        "top_axis_bottom": top_axis_bottom,
        "top_axis_width": top_axis_width,
        "top_axis_height": top_axis_height,
        "left_axis_left": left_axis_left,
        "left_axis_bottom": left_axis_bottom,
        "left_axis_width": left_axis_width,
        "left_axis_height": left_axis_height,
    }


def filter_tick_labels_to_fit(
    labels: list[Artist],
    axis_name: str,
) -> None:
    """Hide only overlapping tick labels."""
    intervals: list[tuple[float, float]] = []
    visible_labels: list[Artist] = []

    for label in labels:
        if not label.get_text():
            label.set_visible(False)
            continue
        label.set_visible(True)
        bbox = label.get_window_extent()
        if axis_name == "x":
            intervals.append((bbox.x0, bbox.x1))
        elif axis_name == "y":
            intervals.append((bbox.y0, bbox.y1))
        else:
            raise ValueError(f"Unsupported label axis: {axis_name}")
        visible_labels.append(label)

    keep_flags = prune_label_intervals(intervals)

    for label, keep_label in zip(visible_labels, keep_flags, strict=True):
        if not keep_label:
            label.set_text("")
            label.set_visible(False)


def prune_clustered_labels(
    figure: plt.Figure,
    matrix_axis: plt.Axes,
    show_all_labels: bool,
) -> None:
    """Hide overlapping clustered labels only when labels are not mandatory."""
    if show_all_labels:
        return
    figure.canvas.draw()
    filter_tick_labels_to_fit(list(matrix_axis.get_xticklabels()), "x")
    filter_tick_labels_to_fit(list(matrix_axis.get_yticklabels()), "y")


def draw_manual_dendrogram(
    axis: plt.Axes,
    dendrogram_info: dict[str, list],
    leaf_count: int,
    orientation: str,
) -> None:
    """Draw a dendrogram whose tips align exactly with matrix cell centres."""
    segments = build_dendrogram_segments(
        dendrogram_info["icoord"],
        dendrogram_info["dcoord"],
        orientation,
    )
    branch_collection = LineCollection(
        segments,
        colors="#4a4a4a",
        linewidths=1.0,
        capstyle="butt",
        joinstyle="miter",
    )
    axis.add_collection(branch_collection)

    max_height = max(max(row) for row in dendrogram_info["dcoord"])
    apply_dendrogram_limits(axis, leaf_count, max_height, orientation)
    style_dendrogram_axis(axis)


def draw_legend(
    axis: plt.Axes,
    cmap: colors.Colormap,
    lower_threshold: float,
    upper_threshold: float,
    compact: bool,
    height_fraction: float,
    top_padding: float,
    include_title: bool = True,
) -> None:
    """Draw a compact boxed legend in its own axis."""
    legend_breaks = build_legend_breaks(lower_threshold, upper_threshold)
    legend_height = min(max(height_fraction, 0.1), 0.95)
    legend_top = min(max(1.0 - top_padding, legend_height), 1.0)
    legend_bottom = legend_top - legend_height
    legend_left = 0.42
    legend_right = 0.78
    rectangle_edges = np.linspace(legend_bottom, legend_top, 257)
    value_positions = np.linspace(0.0, 1.0, 256)

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    for index, value in enumerate(value_positions):
        axis.add_patch(
            Rectangle(
                (legend_left, rectangle_edges[index]),
                legend_right - legend_left,
                rectangle_edges[index + 1] - rectangle_edges[index],
                facecolor=cmap(value),
                edgecolor="none",
            )
        )

    axis.add_patch(
        Rectangle(
            (legend_left, legend_bottom),
            legend_right - legend_left,
            legend_top - legend_bottom,
            facecolor="none",
            edgecolor="#303030",
            linewidth=0.8,
        )
    )

    for break_value in legend_breaks:
        fraction = (break_value - lower_threshold) / (upper_threshold - lower_threshold)
        position = legend_bottom + fraction * legend_height
        axis.plot([legend_left - 0.05, legend_left], [position, position], color="#303030", linewidth=0.8)
        axis.text(
            legend_left - 0.08,
            position,
            f"{break_value:g}",
            ha="right",
            va="center",
            fontsize=6,
            color="#303030",
        )

    if include_title:
        axis.text(
            legend_left - 0.19,
            (legend_top + legend_bottom) / 2,
            "Raw value",
            ha="center",
            va="center",
            rotation=90,
            fontsize=6,
            color="#303030",
        )

def render_simple_figure(
    matrix_values: np.ndarray,
    output_path: Path,
    lower_threshold: float,
    upper_threshold: float,
    colour_palette: str,
) -> None:
    """Render the simple matrix-only figure."""
    cmap = build_colormap(colour_palette)
    norm = colors.Normalize(vmin=lower_threshold, vmax=upper_threshold, clip=True)
    figure = plt.figure(figsize=(derive_device_size(matrix_values.shape[0], simple=True),) * 2)
    grid = GridSpec(
        1,
        2,
        width_ratios=[1.6, 18.0],
        wspace=0.03,
        left=0.04,
        right=0.99,
        bottom=0.05,
        top=0.99,
        figure=figure,
    )

    legend_axis = figure.add_subplot(grid[0, 0])
    matrix_axis = figure.add_subplot(grid[0, 1])

    draw_legend(
        legend_axis,
        cmap,
        lower_threshold,
        upper_threshold,
        compact=True,
        height_fraction=0.26,
        top_padding=0.04,
    )
    draw_simple_matrix(matrix_axis, matrix_values, cmap, norm)

    figure.savefig(output_path, facecolor="white")
    plt.close(figure)


def render_clustered_figure(
    genome_names: list[str],
    matrix_values: np.ndarray,
    output_path: Path,
    matrix_label: str,
    lower_threshold: float,
    upper_threshold: float,
    colour_palette: str,
) -> None:
    """Render the clustered heatmap figure."""
    cmap = build_colormap(colour_palette)
    norm = colors.Normalize(vmin=lower_threshold, vmax=upper_threshold, clip=True)
    condensed_distance = build_distance_condensed(matrix_values)
    linkage_matrix = linkage(condensed_distance, method="complete")
    dendrogram_info = dendrogram(linkage_matrix, no_plot=True)
    ordered_indices = dendrogram_info["leaves"]
    ordered_matrix = matrix_values[np.ix_(ordered_indices, ordered_indices)]
    ordered_names = [genome_names[index] for index in ordered_indices]
    show_all_labels = should_render_sample_labels(len(ordered_names))

    base_dimensions = derive_clustered_base_dimensions(len(ordered_names))
    safety_pad_in = 0.08
    provisional_layout = derive_clustered_layout(
        base_dimensions,
        right_pad_in=base_dimensions["min_right_pad_in"],
        bottom_pad_in=base_dimensions["min_bottom_pad_in"],
    )
    figure = plt.figure(
        figsize=(
            provisional_layout["figure_width_in"],
            provisional_layout["figure_height_in"],
        )
    )
    legend_axis = figure.add_axes(
        [
            provisional_layout["legend_left"],
            provisional_layout["legend_bottom"],
            provisional_layout["legend_width"],
            provisional_layout["legend_height"],
        ]
    )
    matrix_axis = figure.add_axes(
        [
            provisional_layout["matrix_left"],
            provisional_layout["matrix_bottom"],
            provisional_layout["matrix_width"],
            provisional_layout["matrix_height"],
        ]
    )
    draw_legend(
        legend_axis,
        cmap,
        lower_threshold,
        upper_threshold,
        compact=True,
        height_fraction=0.78,
        top_padding=0.08,
        include_title=False,
    )
    draw_matrix(
        matrix_axis,
        ordered_matrix,
        cmap,
        norm,
        ordered_names,
        show_grid=len(ordered_names) <= 75,
        row_labels_right=True,
    )
    figure.canvas.draw()
    final_layout = provisional_layout
    for _ in range(4):
        final_layout, right_overhang_in, bottom_overhang_in = expand_figure_for_labels(
            figure,
            matrix_axis,
            base_dimensions,
            safety_pad_in,
        )
        figure.set_size_inches(
            final_layout["figure_width_in"],
            final_layout["figure_height_in"],
            forward=True,
        )
        legend_axis.set_position(
            [
                final_layout["legend_left"],
                final_layout["legend_bottom"],
                final_layout["legend_width"],
                final_layout["legend_height"],
            ]
        )
        matrix_axis.set_position(
            [
                final_layout["matrix_left"],
                final_layout["matrix_bottom"],
                final_layout["matrix_width"],
                final_layout["matrix_height"],
            ]
        )
        figure.canvas.draw()
        if right_overhang_in <= 0.01 and bottom_overhang_in <= 0.01:
            break
    prune_clustered_labels(figure, matrix_axis, show_all_labels)
    figure.canvas.draw()

    top_axis = figure.add_axes(
        [
            final_layout["top_axis_left"],
            final_layout["top_axis_bottom"],
            final_layout["top_axis_width"],
            final_layout["top_axis_height"],
        ]
    )
    left_axis = figure.add_axes(
        [
            final_layout["left_axis_left"],
            final_layout["left_axis_bottom"],
            final_layout["left_axis_width"],
            final_layout["left_axis_height"],
        ]
    )
    draw_manual_dendrogram(
        top_axis,
        dendrogram_info,
        leaf_count=len(ordered_names),
        orientation="top",
    )
    draw_manual_dendrogram(
        left_axis,
        dendrogram_info,
        leaf_count=len(ordered_names),
        orientation="left",
    )
    figure.canvas.draw()

    figure.savefig(output_path, facecolor="white")
    plt.close(figure)


def write_outputs(
    genome_names: list[str],
    matrix_values: np.ndarray,
    matrix_path: Path,
    lower_threshold: float,
    upper_threshold: float,
    colour_palette: str,
) -> None:
    """Render clustered and simple SVG and PNG outputs beside the input matrix."""
    output_dir = matrix_path.parent
    matrix_label = matrix_path.name
    clustered_svg_path = output_dir / OUTPUT_FILENAMES[0]
    clustered_png_path = output_dir / OUTPUT_FILENAMES[1]
    simple_svg_path = output_dir / OUTPUT_FILENAMES[2]
    simple_png_path = output_dir / OUTPUT_FILENAMES[3]

    render_clustered_figure(
        genome_names,
        matrix_values,
        clustered_svg_path,
        matrix_label,
        lower_threshold,
        upper_threshold,
        colour_palette,
    )
    render_clustered_figure(
        genome_names,
        matrix_values,
        clustered_png_path,
        matrix_label,
        lower_threshold,
        upper_threshold,
        colour_palette,
    )
    render_simple_figure(
        matrix_values,
        simple_svg_path,
        lower_threshold,
        upper_threshold,
        colour_palette,
    )
    render_simple_figure(
        matrix_values,
        simple_png_path,
        lower_threshold,
        upper_threshold,
        colour_palette,
    )

    for output_path in (
        clustered_svg_path,
        simple_svg_path,
        clustered_png_path,
        simple_png_path,
    ):
        print(f"Wrote {output_path}")


def main() -> int:
    """Run the FastAAI matrix visualisation workflow."""
    matplotlib.rcParams["svg.fonttype"] = SVG_FONT_TYPE
    args = parse_args()
    matrix_path = Path(args.matrix_path)
    genome_names, matrix_values = read_matrix_file(matrix_path)
    write_outputs(
        genome_names,
        matrix_values,
        matrix_path,
        args.lower_threshold,
        args.upper_threshold,
        args.colour_palette,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
