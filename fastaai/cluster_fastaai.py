#!/usr/bin/env python3
"""
Cluster genomes from a FastAAI matrix using complete linkage and choose
one representative per cluster.

Required packages: numpy

Core pipeline
-------------
1) Read matrix: parse a FastAAI full tab-separated matrix of FastAAI-coded
   numeric AAI values; enforce strict structure.
2) Validate tables:
   - input_list.tsv: accession, path (1:1, paths must exist).
   - metadata.tsv : required fields present & parseable (see REQUIRED_NONEMPTY_COLS).
   - Name identity: every matrix name must resolve to metadata by plain accession or
     composite alias.
3) Cluster: complete-linkage on distance d = 1 - ANI/100; cut at (1 - threshold)
   so all pairs in a cluster have ANI >= threshold and non-NA (post-checked).
4) Score candidates, per cluster, with channel-specific transforms,
   winsorization (5-95%), and min-max normalization within cluster:
   - A: Assembly level (rank/3; Complete Genome>Chromosome>Scaffold>Contig).
   - B: BUSCO   (Complete - 1*Missing) %, winsorized -> min-max.
   - Q: CheckM2 (Complete - 5*Contamination) %, winsorized -> min-max (Gcode-matched).
   - N: N50, log10(x+1) -> winsorized -> min-max.
   - S: Scaffolds, log10(x+1) -> winsorized -> min-max -> invert (fewer is better).
   - C: ANI centrality within cluster:
           - for each genome i, compute mean ANI to all other members;
           - if max(mean_i) - min(mean_i) < 0.05, treat the cluster as
             homogeneous and set C_i = 0.5 for all members;
           - otherwise rescale the mean ANI values so the minimum becomes 0
             and the maximum becomes 1;
           singleton clusters get C=1.
   Composite score S = wA*A + wB*B + wQ*Q + wN*N + wS*S + wC*C.
   Weight presets chosen by --score-profile {default,isolate,mag}.
5) Select representative: pick highest score (ties if |delta| <= epsilon). Tie-cascade:
   assembly rank (higher) -> BUSCO C (higher) -> BUSCO M (lower) ->
   CheckM2 contamination (lower) -> CheckM2 completeness (higher) -> Scaffolds (fewer) ->
   N50 (higher) -> lexicographically smallest Accession.
6) Soft screen (warn only): list all offenders per cluster where
   Completeness < 90 or Contamination > 5; never hard-fail here.
7) Outputs:
   - cluster.tsv:  Accession, Cluster_ID, Is_Representative, ANI_to_Representative, Score, Path
   - representatives.tsv: Cluster_ID, Representative_Accession, Organism_Name,
     CheckM2_Completeness, CheckM2_Contamination,
     BUSCO, Assembly_Level, N50, Cluster_Size
     (Organism_Name has whitespace collapsed to underscores).

Strict validation:
  - FastAAI matrix names must resolve to metadata accessions using plain accession or
    composite alias matching.
  - Matrix must be square with 'query_genome' in the first header column.
  - Row names must match header names in order.
  - Matrix values must be numeric FastAAI-coded AAI values in [0,100] and symmetric
    within a small tolerance.
  - Thresholds above 90% are not supported because FastAAI matrix output collapses
    all >90% values to 95.0.
  - Gcode must be 4 or 11 (hard fail).
  - Assembly_Level normalized case-insensitively to exactly:
        {'Complete Genome','Chromosome','Scaffold','Contig'} (no synonyms).
  - Required metadata TSV fields must be present & parseable; missing/unparsable => hard fail.
  - accession <-> path must be 1:1 and path must exist on disk.

Threading:
  - --threads is an upper bound; actual worker count is capped by hardware/scheduler limits
    (affinity/cgroups/cgroups v1/v2, SLURM/PBS/SGE). That cap is applied to:
      (a) BLAS/OpenMP/numexpr via env vars
      (b) Python ThreadPoolExecutors.

References:
  - Hierarchical clustering: Murtagh & Contreras, WIREs DMKD (2012), doi:10.1002/widm.53
  - CheckM2: Chklovski et al., Nat Methods (2023), doi:10.1038/s41592-023-01940-w
  - BUSCO v5: Manni et al., Mol Biol Evol (2021), doi:10.1093/molbev/msab199
  - QUAST/N50: Gurevich et al., Bioinformatics (2013), doi:10.1093/bioinformatics/btt086

Author: Akito Shima (asuq)
Email: asuq.4096@gmail.com
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn


# --------------------------------------------------------------------------- #
# CLI, logging, and fatal helper
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cluster genomes from a FastAAI matrix using complete linkage and select "
            "representatives by a composite quality score with deterministic tie-break."
        )
    )
    p.add_argument(
        "-a",
        "--ani-matrix",
        required=True,
        type=Path,
        help=(
            "FastAAI full tab-separated matrix of FastAAI-coded numeric AAI values, "
            "typically FastAAI_matrix.txt."
        ),
    )
    p.add_argument(
        "-l",
        "--input-list",
        required=True,
        type=Path,
        help="TSV with lowercase headers: accession, path.",
    )
    p.add_argument(
        "-m",
        "--metadata",
        required=True,
        type=Path,
        help="TSV with full metadata columns.",
    )
    p.add_argument(
        "-t",
        "--threshold",
        required=False,
        default="0.90",
        help=(
            "AAI threshold as either a fraction in (0,0.90] or a percentage in "
            "(0,90]. Thresholds above 90%% are not applicable to FastAAI matrix "
            "output. Default: 0.90."
        ),
    )
    p.add_argument(
        "-o",
        "--outdir",
        required=True,
        type=Path,
        help="Output directory for cluster.tsv and representatives.tsv.",
    )
    p.add_argument(
        "--cluster-id-prefix",
        default="cluster",
        help="Prefix for generated cluster identifiers. Default: cluster.",
    )
    p.add_argument(
        "-p",
        "--threads",
        required=False,
        type=int,
        default=8,
        help="Upper bound on concurrency; default 8. Capped by hardware/scheduler.",
    )
    p.add_argument(
        "--score-profile",
        choices=["default", "isolate", "mag"],
        default="default",
        help="Weighting profile for representative scoring (default, isolate, mag).",
    )

    # Mutually exclusive logging controls
    log_group = p.add_mutually_exclusive_group()
    log_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Show ERROR, WARNING and CRITICAL only.",
    )
    log_group.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Verbose debug logging (include per-candidate scoring details).",
    )

    return p.parse_args()


def configure_logging(quiet: bool, debug: bool) -> None:
    if quiet:
        level = logging.WARNING
    elif debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


def die(msg: str) -> NoReturn:
    logging.critical(msg)
    sys.exit(1)


def normalise_threshold(raw_threshold: str | float) -> float:
    """
    Normalise the threshold to a fraction in the interval (0, 0.90].
    Accept either fraction form (0.90) or percentage form (90).
    """
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unable to parse threshold value: {raw_threshold}") from exc

    if 0.0 < threshold < 1.0:
        if threshold <= 0.90:
            return threshold
        raise ValueError(
            "Threshold is not applicable to FastAAI matrix output. "
            "FastAAI collapses all >90% AAI values to 95.0 in matrix mode, so "
            "thresholds must be in (0,0.90] or (0,90]. "
            f"Got: {raw_threshold}"
        )
    if 1.0 < threshold <= 90.0:
        return threshold / 100.0
    raise ValueError(
        "Threshold is not applicable to FastAAI matrix output. "
        "FastAAI collapses all >90% AAI values to 95.0 in matrix mode, so "
        "thresholds must be in (0,0.90] or (0,90]. "
        f"Got: {raw_threshold}"
    )


def validate_cluster_id_prefix(prefix: str) -> str:
    """
    Validate the user-supplied cluster identifier prefix.
    """
    cleaned = prefix.strip()
    if not cleaned:
        raise ValueError("Cluster ID prefix must not be empty.")
    if any(character.isspace() for character in cleaned):
        raise ValueError("Cluster ID prefix must not contain whitespace.")
    return cleaned


# --------------------------------------------------------------------------- #
# Scheduler/hardware-aware thread capping
# --------------------------------------------------------------------------- #
def _available_cpus() -> int:
    """
    CPUs effectively available to this process.
    On Linux, sched_getaffinity(0) reflects cpusets/cgroup placement.
    """
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except Exception:
        return os.cpu_count() or 1


def _cgroup_cpu_cap() -> int | None:
    """
    Best-effort CPU cap from cgroups (v2: cpu.max; v1: cpu.cfs_quota_us / cpu.cfs_period_us).
    Returns None if not detectable.
    """
    # cgroups v2
    try:
        cpu_max = Path("/sys/fs/cgroup/cpu.max")
        if cpu_max.is_file():
            txt = cpu_max.read_text().strip().split()
            if len(txt) >= 2 and txt[0] != "max":
                quota = int(txt[0])
                period = int(txt[1])
                if quota > 0 and period > 0:
                    return max(1, quota // period)
    except Exception:
        pass
    # cgroups v1
    try:
        qf = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        pf = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if qf.is_file() and pf.is_file():
            quota = int(qf.read_text().strip())
            period = int(pf.read_text().strip())
            if quota > 0 and period > 0:
                return max(1, quota // period)
    except Exception:
        pass
    return None


def resolve_thread_cap(requested: int) -> int:
    """
    Conservative upper bound on threads: min(requested, scheduler cap, available CPUs),
    also honoring common scheduler env vars.
    """
    candidates: list[int] = [max(1, requested), _available_cpus()]
    for var in ("NSLOTS", "PBS_NUM_PPN", "PBS_NP", "SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK"):
        v = os.environ.get(var)
        if v and v.isdigit():
            candidates.append(int(v))
    # cgroups cap
    cg = _cgroup_cpu_cap()
    if cg is not None:
        candidates.append(cg)
    return max(1, min(candidates))


def set_thread_envs(n_threads: int) -> None:
    """
    Cap math-library threads. If an env var is already set to a positive int, keep min(existing, cap);
    else set to cap.
    """

    def cap(existing: str | None, limit: int) -> str:
        try:
            if existing is not None:
                v = int(existing)
                if v > 0:
                    return str(min(v, limit))
        except Exception:
            pass
        return str(limit)

    keys = [
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",  # macOS Accelerate
        "NUMEXPR_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
    ]
    for k in keys:
        os.environ[k] = cap(os.environ.get(k), n_threads)


# --------------------------------------------------------------------------- #
# Parsing helpers and types
# --------------------------------------------------------------------------- #
REQUIRED_NONEMPTY_COLS: set[str] = {
    "Gcode",
    "N50",
    "Assembly_Level",
    "BUSCO_bacillota_odb12",
    "Scaffolds",
    "Genome_Size",
    "Completeness_gcode4",
    "Completeness_gcode11",
    "Contamination_gcode4",
    "Contamination_gcode11",
}

BUSCO_RE = re.compile(r"C:(?P<C>\d+(?:\.\d+)?)%.*?M:(?P<M>\d+(?:\.\d+)?)%", re.IGNORECASE)

ASSEMBLY_LEVEL_MAP: dict[str, str] = {
    "complete genome": "Complete Genome",
    "chromosome": "Chromosome",
    "scaffold": "Scaffold",
    "contig": "Contig",
}

ASSEMBLY_RANK: dict[str, int] = {
    "Contig": 0,
    "Scaffold": 1,
    "Chromosome": 2,
    "Complete Genome": 3,
}


def normalise_assembly_level(raw: str, *, acc: str) -> str:
    key = raw.strip().casefold()
    if key not in ASSEMBLY_LEVEL_MAP:
        die(f"Unknown Assembly_Level for accession '{acc}': '{raw}'")
    return ASSEMBLY_LEVEL_MAP[key]


def parse_int_like(x: Any, field: str, acc: str) -> int:
    """
    Accept integers or strings that represent integers.
    Float-like string only if exactly integral (e.g., '1234.0').
    """
    try:
        xs = str(x).strip()
        if xs.upper() == "NA" or xs == "":
            die(f"Required integer field '{field}' is empty or 'NA' for accession '{acc}'")
        if re.fullmatch(r"[+-]?\d+", xs):
            return int(xs)
        xf = float(xs)
        if xf.is_integer():
            return int(xf)
        die(f"Non-integer value for '{field}' in accession '{acc}': '{x}'")
    except Exception:
        die(f"Unparsable integer for '{field}' in accession '{acc}': '{x}'")


def parse_float_like(x: Any, field: str, acc: str) -> float:
    try:
        xs = str(x).strip()
        if xs.upper() == "NA" or xs == "":
            die(f"Required float field '{field}' is empty or 'NA' for accession '{acc}'")
        return float(xs)
    except Exception:
        die(f"Unparsable float for '{field}' in accession '{acc}': '{x}'")


def parse_busco(busco_str: str, acc: str) -> tuple[float, float]:
    """
    Extract C and M percentages from a BUSCO string like:
    'C:94.0%[S:93.5%,D:0.5%],F:1.5%,M:4.5%,n:201'
    """
    if not isinstance(busco_str, str) or not busco_str.strip():
        die(f"Missing BUSCO string for accession '{acc}'")
    m = BUSCO_RE.search(busco_str)
    if not m:
        die(f"Unparsable BUSCO string for accession '{acc}': '{busco_str}'")
    return float(m.group("C")), float(m.group("M"))


def sanitise(text: str) -> str:
    """
    Apply the same sanitisation logic as rename_leaf_name.py.
    """
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.replace(".", "_")
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text


def normalise_organism_name_for_alias(organism_name: str) -> str | None:
    """
    Normalise the Organism_Name segment used in composite matrix-label aliases.
    Return None for empty or 'NA' values.
    """
    cleaned = organism_name.strip()
    if not cleaned or cleaned.upper() == "NA":
        return None
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")
    if not cleaned:
        return None
    return cleaned


def resolve_metadata_accession_columns(columns: list[str]) -> list[str]:
    """
    Determine which metadata accession header names are available.
    """
    accession_columns = [column for column in ("accession", "Accession") if column in columns]
    if not accession_columns:
        die("metadata TSV missing accession column. Expected 'accession' or 'Accession'.")
    return accession_columns


def resolve_metadata_accession_value(
    row: dict[str, str],
    accession_columns: list[str],
    row_number: int,
) -> str:
    """
    Resolve a metadata accession value, supporting both header cases.
    """
    values = []
    for column in accession_columns:
        value = str(row.get(column, "")).strip()
        if value:
            values.append(value)
    unique_values = sorted(set(values))
    if len(unique_values) > 1:
        die(
            "metadata TSV accession columns disagree at row "
            f"{row_number}: {unique_values}"
        )
    if not unique_values:
        die(f"metadata TSV accession is empty at row {row_number}")
    accession = unique_values[0]
    if accession.upper() == "NA":
        die(f"metadata TSV accession is 'NA' at row {row_number}")
    return accession


def build_composite_alias(cluster_id: str, accession: str, organism_name: str) -> str:
    """
    Build the composite metadata alias before sanitisation.
    If Organism_Name is empty or 'NA', omit the suffix and keep only
    ${Cluster_ID}_${accession}.
    """
    normalised_organism_name = normalise_organism_name_for_alias(organism_name)
    if normalised_organism_name is None:
        return f"{cluster_id}_{accession}"
    return f"{cluster_id}_{accession}_{normalised_organism_name}"


def add_alias(
    alias_map: dict[str, str],
    alias: str,
    accession: str,
    description: str,
) -> None:
    """
    Add an alias to a mapping and reject collisions across metadata rows.
    """
    existing = alias_map.get(alias)
    if existing is not None and existing != accession:
        die(
            f"Conflicting duplicate {description} '{alias}' for accessions "
            f"'{existing}' and '{accession}'"
        )
    alias_map[alias] = accession


def resolve_matrix_accessions(
    matrix_names: list[str],
    plain_aliases: dict[str, str],
    raw_composite_aliases: dict[str, str],
    sanitised_composite_aliases: dict[str, str],
) -> dict[str, str]:
    """
    Resolve each matrix label to the underlying metadata accession.
    """
    matrix_to_accession: dict[str, str] = {}
    unresolved: list[str] = []
    for matrix_name in matrix_names:
        resolved = plain_aliases.get(matrix_name)
        if resolved is not None:
            matrix_to_accession[matrix_name] = resolved
            continue

        resolved = raw_composite_aliases.get(matrix_name)
        if resolved is not None:
            logging.info(
                "Matrix label '%s' matched metadata accession '%s' via raw composite alias.",
                matrix_name,
                resolved,
            )
            matrix_to_accession[matrix_name] = resolved
            continue

        sanitised_name = sanitise(matrix_name)
        resolved = sanitised_composite_aliases.get(sanitised_name)
        if resolved is not None:
            logging.info(
                "Matrix label '%s' matched metadata accession '%s' via sanitised composite alias '%s'.",
                matrix_name,
                resolved,
                sanitised_name,
            )
            matrix_to_accession[matrix_name] = resolved
            continue

        unresolved.append(matrix_name)

    if unresolved:
        die(
            "Matrix labels must match metadata accessions either directly or via "
            "the composite alias ${Cluster_ID}_${accession}_${Organism_Name} "
            "or ${Cluster_ID}_${accession} when Organism_Name is NA or empty. "
            "The Organism_Name segment is matched in underscored sanitised form. "
            f"Unmatched matrix labels (first 20): {unresolved[:20]}"
        )

    return matrix_to_accession


@dataclass(slots=True)
class Genome:
    Accession: str
    Organism_Name: str
    Gcode: int
    CheckM2_Completeness: float
    CheckM2_Contamination: float
    N50: int
    Scaffolds: int
    Genome_Size: int
    BUSCO_str: str
    BUSCO_C: float
    BUSCO_M: float
    Assembly_Level: str
    Assembly_Rank: int
    Path: str
    Score: float | None = None


def load_matrix(path: Path) -> tuple[list[str], "np.ndarray", dict[str, int]]:
    """
    Load a FastAAI full tab-separated matrix and return:
        - names: list of accession names in matrix order
        - aai:   full symmetric AAI matrix (n x n), float64
        - name_to_idx: mapping from name -> row/column index
    """
    import numpy as np  # local import for type name

    if not path.is_file():
        die(f"AAI matrix file not found: {path}")

    with path.open("rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        rows = [row for row in reader if row]

    if not rows:
        die("AAI matrix is empty.")

    header = rows[0]
    if len(header) < 2:
        die("AAI matrix header must contain 'query_genome' plus at least one genome name.")
    if header[0] != "query_genome":
        die(
            f"AAI matrix first header cell must be 'query_genome', got: '{header[0]}'"
        )

    names = header[1:]
    if len(names) < 2:
        die("AAI matrix must contain at least two genomes.")

    if len(set(names)) != len(names):
        seen: set[str] = set()
        duplicates: list[str] = []
        for name in names:
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(name)
        die(f"AAI matrix header contains duplicate genome names: {duplicates[:10]}")

    matrix_rows = rows[1:]
    if len(matrix_rows) != len(names):
        die(
            f"AAI matrix must be square: expected {len(names)} data rows, found {len(matrix_rows)}."
        )

    ani = np.zeros((len(names), len(names)), dtype=np.float64)
    for row_index, row in enumerate(matrix_rows):
        expected_width = len(names) + 1
        if len(row) != expected_width:
            die(
                f"AAI matrix row {row_index + 1} has {len(row)} columns, expected {expected_width}."
            )

        row_name = row[0]
        expected_name = names[row_index]
        if row_name != expected_name:
            die(
                f"AAI matrix row {row_index + 1} name '{row_name}' does not match header "
                f"name '{expected_name}' at the same position."
            )

        for column_index, raw_value in enumerate(row[1:]):
            try:
                value = float(raw_value)
            except ValueError:
                die(
                    f"Non-numeric AAI value at row '{row_name}', column "
                    f"'{names[column_index]}': '{raw_value}'"
                )
            if not (0.0 <= value <= 100.0):
                die(
                    f"AAI value out of range [0,100] at row '{row_name}', column "
                    f"'{names[column_index]}': {value}"
                )
            ani[row_index, column_index] = value

    if not np.allclose(ani, ani.T, atol=1e-6, rtol=0.0):
        mismatches: list[str] = []
        for row_index in range(len(names)):
            for column_index in range(row_index + 1, len(names)):
                if abs(ani[row_index, column_index] - ani[column_index, row_index]) > 1e-6:
                    mismatches.append(
                        f"{names[row_index]} vs {names[column_index]} "
                        f"({ani[row_index, column_index]} != {ani[column_index, row_index]})"
                    )
                if len(mismatches) >= 5:
                    break
            if len(mismatches) >= 5:
                break
        die(f"AAI matrix is not symmetric (first mismatches): {mismatches}")

    np.fill_diagonal(ani, 100.0)
    logging.info("Loaded AAI matrix with %d taxa.", len(names))
    name_to_idx = {name: index for index, name in enumerate(names)}
    return names, ani, name_to_idx


# --------------------------------------------------------------------------- #
# TSV loading + structural checks
# --------------------------------------------------------------------------- #
def load_and_check_tables(
    input_list: Path,
    metadata: Path,
    matrix_names: list[str],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, str]]:
    """
    Load input_list.tsv and metadata.tsv, perform structural checks:

      - Keep 'NA' as literal strings (not NaN)
      - TSV: required lowercase columns 'accession' and 'path'
      - TSV: accession and path must be unique (1:1 mapping)
      - TSV: every path must exist on disk
      - metadata TSV: required metadata columns must be present
      - Matrix labels resolve to metadata by direct accession or composite alias
      - Resolved metadata accessions must exist in input_list.tsv

    Returns:
        (tsv_by_acc, csv_by_acc, matrix_to_accession) keyed by canonical accession.
    """

    if not input_list.is_file():
        die(f"Input TSV not found: {input_list}")
    if not metadata.is_file():
        die(f"Metadata TSV not found: {metadata}")

    def read_rows(path: Path, delimiter: str) -> tuple[list[dict[str, str]], list[str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None:
                die(f"Table is missing a header row: {path}")
            rows = [
                {key: (value or "") for key, value in row.items()}
                for row in reader
            ]
        return rows, list(reader.fieldnames)

    tsv_rows, tsv_columns = read_rows(input_list, "\t")
    csv_rows, csv_columns = read_rows(metadata, "\t")

    # TSV: required cols and bijection accession <-> path
    for col in ("accession", "path"):
        if col not in tsv_columns:
            die(f"input_list.tsv missing required column: '{col}'")

    tsv_by_acc: dict[str, dict[str, str]] = {}
    seen_paths: dict[str, str] = {}
    dup_acc: list[str] = []
    dup_path: list[str] = []
    for row in tsv_rows:
        accession = row["accession"]
        path_value = row["path"]
        if accession in tsv_by_acc and accession not in dup_acc:
            dup_acc.append(accession)
        else:
            tsv_by_acc[accession] = row
        if path_value in seen_paths and path_value not in dup_path:
            dup_path.append(path_value)
        else:
            seen_paths[path_value] = accession
    if dup_acc:
        die(f"Duplicate Accession(s) in TSV (first few): {dup_acc[:10]}")
    if dup_path:
        die(f"Duplicate Path(s) in TSV (first few): {dup_path[:5]}")

    # Path existence
    for acc, row in tsv_by_acc.items():
        p_str = str(row["path"]).strip()
        if not p_str or p_str.upper() == "NA":
            die(f"Empty or 'NA' path for accession '{acc}' in input_list.tsv")
        if not Path(p_str).exists():
            die(f"Path does not exist for accession '{acc}': {p_str}")

    # Metadata TSV: presence of columns
    accession_columns = resolve_metadata_accession_columns(csv_columns)
    for column in ("Cluster_ID", "Organism_Name"):
        if column not in csv_columns:
            die(f"metadata TSV missing required column: '{column}'")
    missing_cols = [c for c in REQUIRED_NONEMPTY_COLS if c not in csv_columns]
    if missing_cols:
        die(f"Metadata TSV missing required columns: {missing_cols}")

    csv_by_acc: dict[str, dict[str, str]] = {}
    plain_aliases: dict[str, str] = {}
    raw_composite_aliases: dict[str, str] = {}
    sanitised_composite_aliases: dict[str, str] = {}
    for row_number, row in enumerate(csv_rows, start=2):
        accession = resolve_metadata_accession_value(row, accession_columns, row_number)
        if accession in csv_by_acc:
            die(f"Duplicate Accession(s) in metadata TSV (first few): {[accession]}")
        csv_by_acc[accession] = row
        add_alias(plain_aliases, accession, accession, "plain accession alias")

        cluster_id = str(row.get("Cluster_ID", "") or "").strip()
        organism_name = str(row.get("Organism_Name", "") or "").strip()
        if cluster_id and cluster_id.upper() != "NA":
            raw_composite = build_composite_alias(cluster_id, accession, organism_name)
            add_alias(
                raw_composite_aliases,
                raw_composite,
                accession,
                "raw composite alias",
            )
            sanitised_composite = sanitise(raw_composite)
            if sanitised_composite:
                add_alias(
                    sanitised_composite_aliases,
                    sanitised_composite,
                    accession,
                    "sanitised composite alias",
                )

    matrix_to_accession = resolve_matrix_accessions(
        matrix_names=matrix_names,
        plain_aliases=plain_aliases,
        raw_composite_aliases=raw_composite_aliases,
        sanitised_composite_aliases=sanitised_composite_aliases,
    )

    # Identity / coverage: resolved metadata accessions subset of the two TSV inputs (extras warn)
    resolved_accessions = set(matrix_to_accession.values())
    tsv_acc = set(tsv_by_acc)
    csv_acc = set(csv_by_acc)

    missing_in_tsv = sorted(resolved_accessions - tsv_acc)
    if missing_in_tsv:
        die(
            "Resolved metadata accessions must exist in input_list.tsv.\n"
            f"  Missing in input_list.tsv (first 20): {missing_in_tsv[:20]}"
        )

    extras_tsv = sorted(tsv_acc - resolved_accessions)
    extras_csv = sorted(csv_acc - resolved_accessions)
    if extras_tsv:
        logging.warning(
            "Ignoring %d TSV accession(s) not resolved from the AAI matrix (first 20): %s",
            len(extras_tsv),
            extras_tsv[:20],
        )
    if extras_csv:
        logging.warning(
            "Ignoring %d metadata TSV accession(s) not resolved from the AAI matrix (first 20): %s",
            len(extras_csv),
            extras_csv[:20],
        )

    return tsv_by_acc, csv_by_acc, matrix_to_accession


# --------------------------------------------------------------------------- #
# Genome metadata building
# --------------------------------------------------------------------------- #
def build_genome_metadata(
    names: list[str],
    tsv: dict[str, dict[str, str]],
    csv_df: dict[str, dict[str, str]],
    matrix_to_accession: dict[str, str],
) -> dict[str, Genome]:
    """
    Construct per-genome metadata records used for ranking and filters.
    For each matrix label:
      - Enforces required fields to be non-empty/non-'NA'
      - Normalizes Assembly_Level
      - Validates Gcode and selects appropriate CheckM2 Completeness/Contamination fields
      - Parses N50, Scaffolds, Genome_Size, BUSCO, and Path
    """
    meta: dict[str, Genome] = {}
    for matrix_name in names:
        accession = matrix_to_accession[matrix_name]
        if accession not in csv_df or accession not in tsv:
            die(f"Accession '{accession}' missing from input_list.tsv or metadata.tsv after alignment.")

        row = csv_df[accession]

        # Required non-empty columns (not 'NA')
        for col in REQUIRED_NONEMPTY_COLS:
            val = str(row.get(col, "")).strip()
            if val == "" or val.upper() == "NA":
                die(f"Required column '{col}' empty or 'NA' for accession '{accession}'")

        # Normalise Assembly_Level
        asm_level = normalise_assembly_level(str(row["Assembly_Level"]), acc=accession)

        # Gcode & metrics chosen by Gcode
        gcode = parse_int_like(row["Gcode"], "Gcode", accession)
        if gcode not in (4, 11):
            die(f"Gcode must be 4 or 11 for accession '{accession}', got: {gcode}")
        comp_col = "Completeness_gcode4" if gcode == 4 else "Completeness_gcode11"
        cont_col = "Contamination_gcode4" if gcode == 4 else "Contamination_gcode11"

        # Ensure chosen columns exist and non-empty/non-'NA'
        for col in (comp_col, cont_col):
            if col not in row:
                die(f"Metadata TSV missing required column '{col}' for accession '{accession}'")
            val = str(row.get(col, "")).strip()
            if val == "" or val.upper() == "NA":
                die(f"Required column '{col}' empty or 'NA' for accession '{accession}'")

        org_name = str(row.get("Organism_Name", "") or "")
        checkm2 = parse_float_like(row[comp_col], comp_col, accession)
        contam = parse_float_like(row[cont_col], cont_col, accession)
        n50 = parse_int_like(row["N50"], "N50", accession)
        scaffolds = parse_int_like(row["Scaffolds"], "Scaffolds", accession)
        genome_size = parse_int_like(row["Genome_Size"], "Genome_Size", accession)
        busco_str = str(row["BUSCO_bacillota_odb12"])
        busco_c, busco_m = parse_busco(busco_str, accession)
        path = str(tsv[accession]["path"])

        meta[matrix_name] = Genome(
            Accession=matrix_name,
            Organism_Name=org_name,
            Gcode=gcode,
            CheckM2_Completeness=checkm2,
            CheckM2_Contamination=contam,
            N50=n50,
            Scaffolds=scaffolds,
            Genome_Size=genome_size,
            BUSCO_str=busco_str,
            BUSCO_C=busco_c,
            BUSCO_M=busco_m,
            Assembly_Level=asm_level,
            Assembly_Rank=ASSEMBLY_RANK[asm_level],
            Path=path,
        )

    return meta


# --------------------------------------------------------------------------- #
# Clustering + post-check
# --------------------------------------------------------------------------- #
def cluster_complete_linkage(
    ani: "np.ndarray",
    names: list[str],
    threshold: float,
) -> dict[int, list[int]]:
    """
    Complete-linkage clustering on an AAI matrix.
    Merge clusters only when every cross-cluster pair has ANI >= threshold.
    """
    import numpy as np

    threshold_percent = threshold * 100.0
    cluster_lists: list[list[int]] = [[index] for index in range(len(names))]

    def can_merge(left: list[int], right: list[int]) -> bool:
        for left_index in left:
            for right_index in right:
                value = ani[left_index, right_index]
                if np.isnan(value) or value < threshold_percent:
                    return False
        return True

    merged = True
    while merged:
        merged = False
        for left_pos in range(len(cluster_lists)):
            for right_pos in range(left_pos + 1, len(cluster_lists)):
                if can_merge(cluster_lists[left_pos], cluster_lists[right_pos]):
                    cluster_lists[left_pos] = sorted(
                        cluster_lists[left_pos] + cluster_lists[right_pos]
                    )
                    del cluster_lists[right_pos]
                    merged = True
                    break
            if merged:
                break

    clusters: dict[int, list[int]] = {
        label: members for label, members in enumerate(cluster_lists, start=1)
    }

    logging.info(
        "Formed %d complete-linkage clusters at ANI >= %.2f%%.",
        len(clusters),
        threshold_percent,
    )

    # Post-check: every pair in each cluster must be finite ANI >= threshold
    # Checked by the values in the matrix (e.g. ani[0, 1] = ANI distance between sample idx 0 and 1)
    for lab, idxs in clusters.items():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                val = ani[a, b]
                if np.isnan(val) or val < threshold_percent:
                    die(
                        f"Post-check failed: cluster label {lab} contains pair "
                        f"({names[a]}, {names[b]}) with ANI={val} "
                        f"(must be non-NA and >= {threshold_percent:.2f})."
                    )

    return clusters


# --------------------------------------------------------------------------- #
# Representative selection (scoring + tie-break) + cluster IDs
# --------------------------------------------------------------------------- #
def select_representative_for_indices(
    idxs: list[int],
    names: list[str],
    meta: dict[str, Genome],
    ani: "np.ndarray",
    score_profile: str,
) -> tuple[int, str, list[str]]:
    """
    Select one representative by a composite score.

    Scoring components in [0,1]:
      A: Assembly rank / 3
      Q: CheckM2: (completeness - 5*contamination)/100 -> winsorize 5-95% -> min-max
      B: BUSCO:   (BUSCO_C - 1*BUSCO_M)/100 -> winsorize 5-95% -> min-max
      N: N50:     log10(x+1) -> winsorize 5-95% if n>=8 -> min-max
      S: Scaffolds: log10(x+1) -> winsorize 5-95% if n>=8 -> min-max -> invert
      C: ANI centrality within cluster:
           - for each genome i, compute mean ANI to all other members;
           - if max(mean_i) - min(mean_i) < 0.05, treat the cluster as
             homogeneous and set C_i = 0.5 for all members;
           - otherwise rescale the mean ANI values so the minimum becomes 0
             and the maximum becomes 1;
           singleton clusters get C=1.

    Ties (|difference of score| <= _EPS) are broken by:
      1) higher assembly rank,
      2) BUSCO: higher C, then lower M,
      3) CheckM2: lower contamination, then higher completeness
      4) fewer Scaffolds,
      5) higher N50,
      6) lexicographically smallest Accession.

    Weights by profile (sum is not required to be 1):
      - default: A=3.0, Q=2.0, B=1.5, N=0.75, S=0.50, C=0.50
      - isolate: A=3.5, Q=1.5, B=1.5, N=1.00, S=0.25, C=0.50
      - mag    : A=1.0, Q=3.0, B=2.0, N=0.50, S=1.50, C=0.50
    """
    import numpy as np  # local import for type name

    _EPS = 1e-6  # tie tolerance for scores

    dbg: list[str] = []
    accs = [names[i] for i in idxs]
    infos = [meta[a] for a in accs]
    n = len(infos)

    ## Helpers ================================================================
    def winsorize(arr: list[float]) -> np.ndarray:
        """winsorize within cluster at 5-95% only if cluster is reasonably sized"""
        if n < 8:  # guard for tiny clusters
            return np.asarray(arr, dtype=float)
        a = np.asarray(arr, dtype=float)
        ql, qh = np.quantile(a, [0.05, 0.95])
        return np.clip(a, ql, qh)

    def minmax_norm(arr: "np.ndarray") -> "np.ndarray":
        """normalise value with minimum and maximum values"""
        a = arr.astype(float, copy=False)
        lo, hi = float(np.min(a)), float(np.max(a))
        if hi > lo + 1e-12:
            return (a - lo) / (hi - lo)
        return np.full_like(a, 0.5, dtype=float)

    def ani_centrality_within_cluster(
        idxs: list[int],
        ani: "np.ndarray",
        homogeneity_delta: float = 0.05,
    ) -> "np.ndarray":
        """
        Compute ANI centrality within a cluster.

        For each genome i in the cluster:
        - Compute m_i = mean ANI from i to all other members.
        - If max(m_i) - min(m_i) < homogeneity_delta (in % ANI),
            treat the cluster as homogeneous and return 0.5 for all members.
        - Otherwise, linearly scale m_i so that min(m_i) -> 0 and max(m_i) -> 1.
        """
        n = len(idxs)
        if n <= 1:
            return np.ones(n, dtype=float)

        means: list[float] = []
        for i in idxs:
            others = [ani[i, j] for j in idxs if j != i]
            mean_ani = float(np.mean(others)) if others else 100.0
            means.append(mean_ani)

        m = np.asarray(means, dtype=float)
        m_min = float(m.min())
        m_max = float(m.max())
        spread = m_max - m_min

        # Homogeneous cluster: centrality isn't meaningful -> constant 0.5
        if spread < homogeneity_delta:
            return np.full_like(m, 0.5, dtype=float)

        # Heterogeneous cluster: scale to [0, 1]
        return (m - m_min) / spread

    ## Scoring Components =====================================================
    # Assembly
    A_vals = np.array([g.Assembly_Rank / 3.0 for g in infos], dtype=float)

    # BUSCO
    B_raw = np.array([(g.BUSCO_C - 1.0 * g.BUSCO_M) / 100.0 for g in infos], dtype=float)
    B_vals = minmax_norm(winsorize(B_raw))

    # CheckM2
    Q_raw = np.array(
        [(g.CheckM2_Completeness - 5.0 * g.CheckM2_Contamination) / 100.0 for g in infos],
        dtype=float,
    )
    Q_vals = minmax_norm(winsorize(Q_raw))

    # N50
    n50_log = np.log10(np.array([g.N50 for g in infos], dtype=float) + 1.0)
    N_vals = minmax_norm(winsorize(n50_log))

    # Scaffolds
    cont_log = np.log10(np.array([g.Scaffolds for g in infos], dtype=float) + 1.0)
    S_vals = 1.0 - minmax_norm(winsorize(cont_log))

    # ANI centrality
    C_vals = ani_centrality_within_cluster(idxs, ani, homogeneity_delta=0.05)

    ## Weights ================================================================
    if score_profile == "isolate":
        wA, wQ, wB, wN, wS, wC = 3.5, 1.5, 1.5, 1.00, 0.25, 0.50
    elif score_profile == "mag":
        wA, wQ, wB, wN, wS, wC = 1.0, 3.0, 2.0, 0.50, 1.50, 0.50
    else:  # default
        wA, wQ, wB, wN, wS, wC = 3.0, 2.0, 1.5, 0.75, 0.50, 0.50

    ## Score per candidate ====================================================

    # Compute normalized total score and collect component breakdowns for DEBUG
    scored: list[tuple[str, float, dict[str, float], Genome]] = []
    for i_loc, g in enumerate(infos):
        comps = {
            "A": float(A_vals[i_loc]),
            "Q": float(Q_vals[i_loc]),
            "B": float(B_vals[i_loc]),
            "N": float(N_vals[i_loc]),
            "S": float(S_vals[i_loc]),
            "C": float(C_vals[i_loc]),
            "Q_raw": float(Q_raw[i_loc]),
            "B_raw": float(B_raw[i_loc]),
        }
        total = (
            wA * comps["A"]
            + wQ * comps["Q"]
            + wB * comps["B"]
            + wN * comps["N"]
            + wS * comps["S"]
            + wC * comps["C"]
        )

        g.Score = float(total)

        scored.append((g.Accession, float(total), comps, g))

    # Emit per-candidate debug lines
    dbg.append(
        f"Cluster size={n}; profile={score_profile}; weights:"
        f" A={wA}, Q={wQ}, B={wB}, N={wN}, S={wS}, C={wC}"
    )
    for acc, total, c, g in sorted(scored, key=lambda x: (-x[1], x[0])):
        dbg.append(
            "  %s: score=%.6f | A=%.3f Q=%.3f(q_raw=%.3f) B=%.3f(b_raw=%.3f)"
            " N=%.3f S=%.3f C=%.3f | asm=%s(rank=%d) busco(C=%.2f,M=%.2f)"
            " checkm2=%.2f contam=%.2f scaffolds=%d n50=%d"
            % (
                acc,
                total,
                c["A"],
                c["Q"],
                c["Q_raw"],
                c["B"],
                c["B_raw"],
                c["N"],
                c["S"],
                c["C"],
                g.Assembly_Level,
                g.Assembly_Rank,
                g.BUSCO_C,
                g.BUSCO_M,
                g.CheckM2_Completeness,
                g.CheckM2_Contamination,
                g.Scaffolds,
                g.N50,
            )
        )

    # Primary selection by score
    scored.sort(key=lambda x: (-x[1], x[0]))
    top_score = scored[0][1]
    tied = [acc for acc, s, _, _ in scored if abs(s - top_score) <= _EPS]
    if len(tied) == 1:
        dbg.append(f"Chosen by score alone: {tied[0]} (score={top_score:.6f})")
        return n, tied[0], dbg

    ## Tie cascade ============================================================
    acc2gen = {g.Accession: g for _, _, _, g in scored}

    def reduce_tie(
        acc_list: list[str], key_fn, desc: str, prefer_lower: bool = False
    ) -> list[str]:
        if len(acc_list) <= 1:
            return acc_list
        vals = [key_fn(acc2gen[a]) for a in acc_list]
        best = (min if prefer_lower else max)(vals)
        winners = [a for a, v in zip(acc_list, vals, strict=True) if v == best]
        if len(winners) < len(acc_list):
            dbg.append(f"Tie-break by {desc}: {len(acc_list)} -> {len(winners)} (best={best}).")
        return winners

    # 1) higher assembly rank
    # 2) BUSCO: higher C, then lower M
    # 3) CheckM2: lower contamination, then higher completeness
    # 4) fewer Scaffolds
    # 5) higher N50
    # 6) lexicographically smallest accession
    tied = reduce_tie(tied, lambda g: g.Assembly_Rank, "Assembly rank (higher)")
    tied = reduce_tie(tied, lambda g: g.BUSCO_C, "BUSCO C (higher)")
    tied = reduce_tie(tied, lambda g: g.BUSCO_M, "BUSCO M (lower)", True)
    tied = reduce_tie(tied, lambda g: g.CheckM2_Contamination, "Contamination (lower)", True)
    tied = reduce_tie(tied, lambda g: g.CheckM2_Completeness, "CheckM2 (higher)")
    tied = reduce_tie(tied, lambda g: g.Scaffolds, "Scaffolds (fewer)", True)
    tied = reduce_tie(tied, lambda g: g.N50, "N50 (higher)")

    rep_acc = sorted(tied)[0]
    if len(tied) > 1:
        dbg.append(
            f"Final tie among {len(tied)} candidate(s); picking lexicographically smallest: {rep_acc}"
        )
    else:
        dbg.append(f"Chosen by tie cascade: {rep_acc}")

    return n, rep_acc, dbg


def select_representatives_for_clusters(
    clusters: dict[int, list[int]],
    names: list[str],
    meta: dict[str, Genome],
    ani: "np.ndarray",
    score_profile: str,
    threads: int,
) -> list[tuple[int, str, int, list[int], list[str]]]:
    """
    Select representatives for all clusters in parallel.

    Returns:
        A list of tuples:
          (cluster_size, rep_acc, original_label, idxs, debug_lines)
    """
    results: list[tuple[int, str, int, list[int], list[str]]] = []
    with ThreadPoolExecutor(max_workers=threads) as ex:
        fut_to_lab = {
            ex.submit(
                select_representative_for_indices,
                idxs,
                names,
                meta,
                ani,
                score_profile,
            ): lab
            for lab, idxs in clusters.items()
        }
        for fut in as_completed(fut_to_lab):
            lab = fut_to_lab[fut]
            try:
                size, rep_acc, dbg_lines = fut.result()
                results.append((size, rep_acc, lab, clusters[lab], dbg_lines))
            except Exception as e:
                die(f"Error selecting representative for cluster {lab}: {e}")
    return results


def assign_cluster_ids(
    results: list[tuple[int, str, int, list[int], list[str]]],
    prefix: str,
) -> tuple[dict[str, str], dict[str, list[int]]]:
    """
    Assign stable Cluster_IDs (<prefix>1, <prefix>2, ...) to clusters.
    Clusters are ordered by:
      - size (descending)
      - representative accession (lex ascending)
    """
    results.sort(key=lambda x: (-x[0], x[1]))
    rep_by_cid: dict[str, str] = {}  # cluster idx = representative sample idx
    idxs_by_cid: dict[str, list[int]] = {}  # cluster idx = samples idx in cluster
    for i, (size, rep_acc, _lab, idxs, dbg_lines) in enumerate(results, start=1):
        cid = f"{prefix}{i}"
        rep_by_cid[cid] = rep_acc
        idxs_by_cid[cid] = idxs
        for line in dbg_lines:
            logging.debug("%s (%s): %s", cid, rep_acc, line)
    return rep_by_cid, idxs_by_cid


def warn_gcode_mixture(
    idxs_by_cid: dict[str, list[int]],
    names: list[str],
    meta: dict[str, Genome],
) -> None:
    """
    Emit warnings for clusters that mix Gcode 4 and 11 genomes.
    """
    for cid, idxs in idxs_by_cid.items():
        g4 = g11 = 0
        for idx in idxs:
            g = meta[names[idx]].Gcode
            if g == 4:
                g4 += 1
            elif g == 11:
                g11 += 1
        if g4 > 0 and g11 > 0:
            logging.warning("Gcode mixture in %s: gcode4=%d, gcode11=%d", cid, g4, g11)


def warn_soft_screen_offenders(
    idxs_by_cid: dict[str, list[int]],
    names: list[str],
    meta: dict[str, Genome],
) -> None:
    """
    List all soft-screen offenders per cluster (warn only).
    Offender: Completeness < 90 or Contamination > 5.
    """
    for cid, idxs in idxs_by_cid.items():
        offenders: list[str] = []
        for idx in idxs:
            g = meta[names[idx]]
            reasons: list[str] = []
            if g.CheckM2_Completeness < 90:
                reasons.append(f"Completeness={g.CheckM2_Completeness:.2f}")
            if g.CheckM2_Contamination > 5:
                reasons.append(f"Contam={g.CheckM2_Contamination:.2f}%")
            if reasons:
                offenders.append(f"{g.Accession}[{';'.join(reasons)}]")
        if offenders:
            logging.warning("Soft-screen offenders in %s: %s", cid, ", ".join(offenders))


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def build_cluster_rows(
    rep_by_cid: dict[str, str],
    idxs_by_cid: dict[str, list[int]],
    names: list[str],
    ani: "np.ndarray",
    name_to_idx: dict[str, int],
    meta: dict[str, Genome],
) -> list[tuple[str, str, str, str, str, str]]:
    """
    Construct rows for cluster.tsv.
    Each row has:
        Accession, Cluster_ID, Is_Representative, ANI_to_Representative, Score, Path
    """
    import numpy as np  # local import for type name

    rows: list[tuple[str, str, str, str, str, str]] = []
    for cid, idxs in idxs_by_cid.items():
        rep_acc = rep_by_cid[cid]
        rep_idx = name_to_idx[rep_acc]
        for idx in sorted(idxs, key=lambda k: names[k]):
            acc = names[idx]
            path = meta[acc].Path
            score = meta[acc].Score

            is_rep = "yes" if acc == rep_acc else "no"
            if acc == rep_acc:
                ani_to_rep = "100.0000"
            else:
                v = ani[idx, rep_idx]
                if np.isnan(v):
                    die(
                        f"Internal error: NA ANI inside complete-link cluster {cid} "
                        f"between '{acc}' and representative '{rep_acc}'"
                    )
                ani_to_rep = f"{v:.4f}"

            if score is None:
                die(
                    f"Internal error: composite score not set for accession '{acc}' "
                    f"in cluster {cid}"
                )
            score_str = f"{score:.6f}"

            rows.append((acc, cid, is_rep, ani_to_rep, score_str, path))
    return rows


def build_representative_rows(
    rep_by_cid: dict[str, str],
    idxs_by_cid: dict[str, list[int]],
    meta: dict[str, Genome],
) -> list[tuple[str, str, str, str, str, str, str, str, str]]:
    """
    Construct rows for representatives.tsv.
    Each row has:
        Cluster_ID, Representative_Accession, Organism_Name,
        CheckM2_Completeness, CheckM2_Contamination
        BUSCO, Assembly_Level, N50, Cluster_Size
    """
    rows: list[tuple[str, str, str, str, str, str, str, str, str]] = []
    for cid, idxs in idxs_by_cid.items():
        rep_acc = rep_by_cid[cid]
        g = meta[rep_acc]
        org_out = re.sub(r"\s+", "_", (g.Organism_Name or "").strip())
        rows.append(
            (
                cid,
                rep_acc,
                org_out,
                f"{g.CheckM2_Completeness:.2f}",
                f"{g.CheckM2_Contamination:.2f}",
                g.BUSCO_str,
                g.Assembly_Level,
                f"{g.N50:d}",
                str(len(idxs)),
            )
        )
    return rows


def write_outputs(
    outdir: Path,
    cluster_rows: list[tuple[str, str, str, str, str, str]],
    reps_rows: list[tuple[str, str, str, str, str, str, str, str, str]],
) -> None:
    """
    Write cluster.tsv and representatives.tsv to the output directory.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    cluster_tsv = outdir / "cluster.tsv"
    reps_tsv = outdir / "representatives.tsv"

    with cluster_tsv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(
            [
                "Accession",
                "Cluster_ID",
                "Is_Representative",
                "ANI_to_Representative",
                "Score",
                "Path",
            ]
        )
        w.writerows(cluster_rows)

    with reps_tsv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(
            [
                "Cluster_ID",
                "Representative_Accession",
                "Organism_Name",
                "CheckM2_Completeness",
                "CheckM2_Contamination",
                "BUSCO",
                "Assembly_Level",
                "N50",
                "Cluster_Size",
            ]
        )
        w.writerows(reps_rows)

    logging.info("Wrote: %s", cluster_tsv)
    logging.info("Wrote: %s", reps_tsv)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_pipeline(args: argparse.Namespace, threads: int) -> None:
    """
    Run the full clustering and representative selection pipeline.

    Steps:
      1) Load AAI matrix.
      2) Load and validate TSV tables.
      3) Build per-genome metadata.
      4) Complete-linkage clustering.
      5) Parallel representative selection.
      6) Assign cluster IDs.
      7) Warn on Gcode mixtures.
      8) Build output rows.
      9) Write output files.
    """
    try:
        threshold = normalise_threshold(args.threshold)
    except ValueError as exc:
        die(str(exc))

    try:
        cluster_id_prefix = validate_cluster_id_prefix(args.cluster_id_prefix)
    except ValueError as exc:
        die(str(exc))

    # 1) AAI matrix
    names, ani, name_to_idx = load_matrix(args.ani_matrix)

    # 2) Tables + structural checks
    tsv, csv_df, matrix_to_accession = load_and_check_tables(
        input_list=args.input_list,
        metadata=args.metadata,
        matrix_names=names,
    )

    # 3) Build per-genome metadata
    meta = build_genome_metadata(names, tsv, csv_df, matrix_to_accession)

    # 4) Complete-linkage clustering
    clusters = cluster_complete_linkage(ani, names, threshold)

    # 5) Representative selection (parallel)
    results = select_representatives_for_clusters(
        clusters=clusters,
        names=names,
        meta=meta,
        ani=ani,
        score_profile=args.score_profile,
        threads=threads,
    )

    # 6) Assign cluster IDs
    rep_by_cid, idxs_by_cid = assign_cluster_ids(results, cluster_id_prefix)

    # 7) Warnings
    warn_gcode_mixture(idxs_by_cid, names, meta)
    warn_soft_screen_offenders(idxs_by_cid, names, meta)

    # 8) Build output rows
    cluster_rows = build_cluster_rows(
        rep_by_cid=rep_by_cid,
        idxs_by_cid=idxs_by_cid,
        names=names,
        ani=ani,
        name_to_idx=name_to_idx,
        meta=meta,
    )
    reps_rows = build_representative_rows(
        rep_by_cid=rep_by_cid,
        idxs_by_cid=idxs_by_cid,
        meta=meta,
    )

    # 9) Write outputs
    write_outputs(args.outdir, cluster_rows, reps_rows)


def main() -> int:
    """
    Entry point: parse arguments, configure logging and threading, then run pipeline.
    """
    args = parse_args()
    configure_logging(args.quiet, args.debug)

    # Threads: respect both user request and hardware/scheduler limits
    requested = max(1, int(args.threads))
    threads = resolve_thread_cap(requested)
    if threads < requested:
        logging.info(
            "Capping --threads from %d to %d based on hardware/scheduler limits.",
            requested,
            threads,
        )

    # Cap BLAS/OpenMP/numexpr before heavy imports:
    set_thread_envs(threads)

    run_pipeline(args, threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
