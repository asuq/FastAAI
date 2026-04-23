"""Generate persistent FastAAI visualiser test matrices and figures."""

from __future__ import annotations

import shutil
import subprocess
import sys
from math import floor
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PATH = Path(sys.executable)
SCRIPT_PATH = REPO_ROOT / "scripts" / "visualise_aai_matrix.py"
DATA_DIR = REPO_ROOT / "tests" / "data"
FIGURES_DIR = REPO_ROOT / "tests" / "figures"
FULL_MATRIX_PATH = DATA_DIR / "matrix.tsv"
MATRIX_20_PATH = DATA_DIR / "matrix_20.tsv"
MATRIX_100_PATH = DATA_DIR / "matrix_100.tsv"
EXPECTED_OUTPUTS = (
    "FastAAI_matrix_heatmap.svg",
    "FastAAI_matrix_heatmap.png",
    "FastAAI_matrix_heatmap_simple.svg",
    "FastAAI_matrix_heatmap_simple.png",
)
RENDER_TARGETS = (
    ("matrix_20", MATRIX_20_PATH),
    ("matrix_100", MATRIX_100_PATH),
    ("matrix_full", FULL_MATRIX_PATH),
)
TEST_ENV = {
    **os.environ,
    "MPLCONFIGDIR": "/tmp/fastaai_mplconfig",
}


def load_matrix_rows(matrix_path: Path) -> list[list[str]]:
    """Load a FastAAI matrix fixture as tab-separated rows."""
    return [
        line.rstrip("\n").split("\t")
        for line in matrix_path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def evenly_spaced_indices(total_size: int, subset_size: int) -> list[int]:
    """Return deterministic evenly spaced zero-based indices across a range."""
    if subset_size <= 0 or subset_size > total_size:
        raise ValueError("subset_size must be between 1 and total_size")
    if subset_size == 1:
        return [0]
    indices = {
        min(total_size - 1, floor(index * (total_size - 1) / (subset_size - 1)))
        for index in range(subset_size)
    }
    return sorted(indices)


def write_submatrix(
    matrix_rows: list[list[str]],
    data_indices: list[int],
    destination_path: Path,
) -> None:
    """Write a square FastAAI submatrix to a stable fixture path."""
    header = matrix_rows[0]
    selected_header = [header[0]] + [header[index + 1] for index in data_indices]
    output_rows = ["\t".join(selected_header)]
    for row_index in data_indices:
        row = matrix_rows[row_index + 1]
        selected_row = [row[0]] + [row[index + 1] for index in data_indices]
        output_rows.append("\t".join(selected_row))
    destination_path.write_text("\n".join(output_rows) + "\n", encoding="utf-8")


def refresh_derived_matrices() -> None:
    """Regenerate the committed 20- and 100-sample matrix fixtures."""
    matrix_rows = load_matrix_rows(FULL_MATRIX_PATH)
    matrix_size = len(matrix_rows) - 1
    write_submatrix(
        matrix_rows,
        evenly_spaced_indices(matrix_size, 20),
        MATRIX_20_PATH,
    )
    write_submatrix(
        matrix_rows,
        evenly_spaced_indices(matrix_size, 100),
        MATRIX_100_PATH,
    )


def prepare_render_directory(output_dir: Path) -> None:
    """Create a clean render directory for stable output files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for output_name in ("FastAAI_matrix.txt", *EXPECTED_OUTPUTS):
        output_path = output_dir / output_name
        output_path.unlink(missing_ok=True)


def run_visualiser(matrix_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the Python visualiser against a matrix file."""
    return subprocess.run(
        [str(PYTHON_PATH), str(SCRIPT_PATH), str(matrix_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=TEST_ENV,
    )


def render_fixture(source_matrix_path: Path, output_dir: Path) -> None:
    """Copy a matrix into a stable figure directory and render all outputs there."""
    prepare_render_directory(output_dir)
    render_input_path = output_dir / "FastAAI_matrix.txt"
    shutil.copy2(source_matrix_path, render_input_path)

    result = run_visualiser(render_input_path)
    if result.returncode != 0:
        raise RuntimeError(
            "\n".join(
                [
                    f"Failed to render figures for {source_matrix_path}",
                    result.stdout.strip(),
                    result.stderr.strip(),
                ]
            ).strip()
        )

    missing_outputs = [
        output_name
        for output_name in EXPECTED_OUTPUTS
        if not (output_dir / output_name).is_file() or (output_dir / output_name).stat().st_size == 0
    ]
    if missing_outputs:
        raise RuntimeError(
            f"Missing or empty outputs in {output_dir}: {', '.join(missing_outputs)}"
        )


def refresh_rendered_figures() -> None:
    """Render figures for the derived and full matrix fixtures under tests/figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for output_name, source_matrix_path in RENDER_TARGETS:
        render_fixture(source_matrix_path, FIGURES_DIR / output_name)


def main() -> int:
    """Regenerate visualiser test matrices and figures."""
    if not FULL_MATRIX_PATH.is_file():
        raise FileNotFoundError(f"Missing source matrix fixture: {FULL_MATRIX_PATH}")

    refresh_derived_matrices()
    refresh_rendered_figures()
    return 0


if __name__ == "__main__":
    sys.exit(main())
