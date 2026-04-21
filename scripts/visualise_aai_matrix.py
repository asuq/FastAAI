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
import seaborn as sns
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


EXPECTED_HEADER = "query_genome"
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
        default=30.0,
        help="Lower heatmap threshold for colour clipping.",
    )
    parser.add_argument(
        "--upper-threshold",
        type=float,
        default=90.0,
        help="Upper heatmap threshold for colour clipping.",
    )
    args = parser.parse_args()
    validate_thresholds(args.lower_threshold, args.upper_threshold)
    return args


def validate_thresholds(lower_threshold: float, upper_threshold: float) -> None:
    """Validate the user-supplied threshold range."""
    if not 0 <= lower_threshold <= 95:
        die(f"Lower threshold must be within [0,95], got: {lower_threshold}")
    if not 0 <= upper_threshold <= 95:
        die(f"Upper threshold must be within [0,95], got: {upper_threshold}")
    if lower_threshold >= upper_threshold:
        die(
            "Lower threshold must be smaller than upper threshold, "
            f"got: {lower_threshold} >= {upper_threshold}"
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
            if value < 0 or value > 95:
                die(
                    "FastAAI value out of the supported raw range [0,95] at row "
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
    """Convert the observed FastAAI similarity scale into condensed distances."""
    similarity_ceiling = float(matrix_values.max())
    distance_values = similarity_ceiling - matrix_values
    distance_values[(distance_values < 0) & (distance_values > -SYMMETRY_TOLERANCE)] = 0
    if np.any(distance_values < 0):
        die("Derived negative distances from the FastAAI matrix.")
    np.fill_diagonal(distance_values, 0)
    return squareform(distance_values, checks=False)


def derive_label_stride(genome_count: int) -> int:
    """Return the deterministic axis-label thinning interval."""
    if genome_count <= 60:
        return 1
    if genome_count <= 150:
        return 5
    if genome_count <= 400:
        return 10
    return 25


def build_axis_labels(genome_names: list[str]) -> list[str]:
    """Keep a sparse and deterministic subset of axis labels."""
    genome_count = len(genome_names)
    stride = derive_label_stride(genome_count)
    keep_indices = set(range(0, genome_count, stride))
    keep_indices.add(genome_count - 1)
    return [
        genome_name if index in keep_indices else ""
        for index, genome_name in enumerate(genome_names)
    ]


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


def build_colormap() -> colors.LinearSegmentedColormap:
    """Build the shared FastAAI heatmap palette."""
    palette = sns.blend_palette(
        ["#f7fbff", "#dbe9f6", "#9ecae1", "#4292c6", "#2171b5", "#084594"],
        n_colors=256,
        as_cmap=False,
    )
    return colors.LinearSegmentedColormap.from_list("fastaai_heatmap", palette)


def build_legend_breaks(lower_threshold: float, upper_threshold: float) -> list[float]:
    """Return the compact legend break points."""
    midpoint = round((lower_threshold + upper_threshold) / 2, 2)
    return sorted({lower_threshold, midpoint, upper_threshold})


def get_heatmap_extent(leaf_count: int) -> tuple[float, float, float, float]:
    """Return the explicit image extent for a square heatmap."""
    return (-0.5, leaf_count - 0.5, leaf_count - 0.5, -0.5)


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
) -> None:
    """Draw a compact boxed legend in its own axis."""
    legend_breaks = build_legend_breaks(lower_threshold, upper_threshold)
    legend_height = min(max(height_fraction, 0.1), 0.95)
    legend_top = min(max(1.0 - top_padding, legend_height), 1.0)
    legend_bottom = legend_top - legend_height
    legend_left = 0.48
    legend_right = 0.72
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
) -> None:
    """Render the simple matrix-only figure."""
    cmap = build_colormap()
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
) -> None:
    """Render the clustered heatmap figure."""
    cmap = build_colormap()
    norm = colors.Normalize(vmin=lower_threshold, vmax=upper_threshold, clip=True)
    condensed_distance = build_distance_condensed(matrix_values)
    linkage_matrix = linkage(condensed_distance, method="complete")
    dendrogram_info = dendrogram(linkage_matrix, no_plot=True)
    ordered_indices = dendrogram_info["leaves"]
    ordered_matrix = matrix_values[np.ix_(ordered_indices, ordered_indices)]
    ordered_names = [genome_names[index] for index in ordered_indices]

    figure = plt.figure(figsize=(derive_device_size(matrix_values.shape[0], simple=False),) * 2)

    legend_left = 0.04
    legend_width = 0.05
    legend_gap = 0.02
    left_dendrogram_width = 0.05
    right_label_space = 0.16
    bottom_label_space = 0.08
    top_dendrogram_height = 0.12
    top_margin = 0.03

    matrix_left = legend_left + legend_width + legend_gap + left_dendrogram_width
    max_matrix_width = 1.0 - right_label_space - matrix_left
    max_matrix_height = 1.0 - top_margin - top_dendrogram_height - bottom_label_space
    matrix_size = min(max_matrix_width, max_matrix_height)

    legend_axis = figure.add_axes(
        [legend_left, bottom_label_space, legend_width, matrix_size]
    )
    matrix_axis = figure.add_axes(
        [matrix_left, bottom_label_space, matrix_size, matrix_size]
    )
    draw_legend(
        legend_axis,
        cmap,
        lower_threshold,
        upper_threshold,
        compact=True,
        height_fraction=0.28,
        top_padding=0.04,
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
    matrix_position = matrix_axis.get_position()

    top_axis = figure.add_axes(
        [
            matrix_position.x0,
            matrix_position.y1,
            matrix_position.width,
            top_dendrogram_height,
        ]
    )
    left_axis = figure.add_axes(
        [
            matrix_position.x0 - left_dendrogram_width,
            matrix_position.y0,
            left_dendrogram_width,
            matrix_position.height,
        ]
    )
    draw_manual_dendrogram(
        top_axis,
        dendrogram_info,
        leaf_count=len(ordered_names),
        orientation="top",
    )
    top_axis.set_title(matrix_label, fontsize=7.5, pad=2)
    draw_manual_dendrogram(
        left_axis,
        dendrogram_info,
        leaf_count=len(ordered_names),
        orientation="left",
    )

    figure.savefig(output_path, facecolor="white")
    plt.close(figure)


def write_outputs(
    genome_names: list[str],
    matrix_values: np.ndarray,
    matrix_path: Path,
    lower_threshold: float,
    upper_threshold: float,
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
    )
    render_clustered_figure(
        genome_names,
        matrix_values,
        clustered_png_path,
        matrix_label,
        lower_threshold,
        upper_threshold,
    )
    render_simple_figure(matrix_values, simple_svg_path, lower_threshold, upper_threshold)
    render_simple_figure(matrix_values, simple_png_path, lower_threshold, upper_threshold)

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
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
