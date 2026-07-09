"""Tests for the FastAAI clustering helper."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "fastaai" / "cluster_fastaai.py"
MODULE_SPEC = importlib.util.spec_from_file_location("test_cluster_fastaai_module", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load test module from {MODULE_PATH}")
CLUSTER_FASTAAI = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = CLUSTER_FASTAAI
MODULE_SPEC.loader.exec_module(CLUSTER_FASTAAI)

assign_cluster_ids = CLUSTER_FASTAAI.assign_cluster_ids
cluster_hierarchical = CLUSTER_FASTAAI.cluster_hierarchical
Genome = CLUSTER_FASTAAI.Genome
load_matrix = CLUSTER_FASTAAI.load_matrix
load_and_check_tables = CLUSTER_FASTAAI.load_and_check_tables
normalise_threshold = CLUSTER_FASTAAI.normalise_threshold
parse_args = CLUSTER_FASTAAI.parse_args
run_pipeline = CLUSTER_FASTAAI.run_pipeline
sanitise = CLUSTER_FASTAAI.sanitise
taxonomic_level_for_threshold = CLUSTER_FASTAAI.taxonomic_level_for_threshold
normalise_organism_name_for_alias = CLUSTER_FASTAAI.normalise_organism_name_for_alias
select_representative_for_indices = CLUSTER_FASTAAI.select_representative_for_indices


BUSCO = "C:98.0%[S:97.0%,D:1.0%],F:1.0%,M:1.0%,n:200"


def write_text(path: Path, content: str) -> None:
    """Write ASCII test content to a file."""
    path.write_text(content, encoding="ascii")


class ClusterFastAAITests(unittest.TestCase):
    """Verify FastAAI matrix parsing and clustering behaviour."""

    def write_metadata_tsv(
        self,
        path: Path,
        rows: list[dict[str, str]],
        *,
        accession_header: str = "Accession",
    ) -> None:
        """Write a metadata TSV fixture with a configurable accession header."""
        fieldnames = [
            accession_header,
            "Cluster_ID",
            "Assembly_Name",
            "Organism_Name",
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
        ]
        with path.open("w", encoding="ascii", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    def metadata_row(
        self,
        accession: str,
        cluster_id: str,
        organism_name: str,
        *,
        accession_header: str = "Accession",
    ) -> dict[str, str]:
        """Build a minimal metadata row fixture."""
        return {
            accession_header: accession,
            "Cluster_ID": cluster_id,
            "Assembly_Name": f"{accession}_asm",
            "Organism_Name": organism_name,
            "Gcode": "11",
            "N50": "100000",
            "Assembly_Level": "Scaffold",
            "BUSCO_bacillota_odb12": BUSCO,
            "Scaffolds": "10",
            "Genome_Size": "4000000",
            "Completeness_gcode4": "98.0",
            "Completeness_gcode11": "98.0",
            "Contamination_gcode4": "0.5",
            "Contamination_gcode11": "0.5",
        }

    def genome(
        self,
        accession: str,
        *,
        assembly_level: str = "Scaffold",
        assembly_rank: int = 1,
        checkm2_completeness: float = 98.0,
        checkm2_contamination: float = 0.5,
        n50: int = 100000,
        scaffolds: int = 10,
        busco_c: float = 98.0,
        busco_m: float = 1.0,
    ) -> Genome:
        """Build a minimal Genome fixture for representative scoring tests."""
        return Genome(
            Accession=accession,
            Organism_Name=f"Organism_{accession}",
            Gcode=11,
            CheckM2_Completeness=checkm2_completeness,
            CheckM2_Contamination=checkm2_contamination,
            N50=n50,
            Scaffolds=scaffolds,
            Genome_Size=4000000,
            BUSCO_str=BUSCO,
            BUSCO_C=busco_c,
            BUSCO_M=busco_m,
            Assembly_Level=assembly_level,
            Assembly_Rank=assembly_rank,
            Path=f"/tmp/{accession}.fna",
        )

    def test_load_matrix_accepts_valid_square_matrix(self) -> None:
        """Load a valid FastAAI matrix, log AAI wording, and normalise the diagonal."""
        with tempfile.TemporaryDirectory() as tempdir:
            matrix_path = Path(tempdir) / "FastAAI_matrix.txt"
            write_text(
                matrix_path,
                "\n".join(
                    [
                        "query_genome\tA\tB\tC",
                        "A\t95\t97\t96",
                        "B\t97\t95\t95.5",
                        "C\t96\t95.5\t95",
                    ]
                )
                + "\n",
            )

            with self.assertLogs(level="INFO") as captured:
                names, ani, name_to_idx = load_matrix(matrix_path)

            self.assertEqual(names, ["A", "B", "C"])
            self.assertEqual(name_to_idx["B"], 1)
            self.assertAlmostEqual(ani[0, 1], 97.0)
            self.assertAlmostEqual(ani[1, 2], 95.5)
            self.assertAlmostEqual(ani[0, 0], 100.0)
            self.assertIn("Loaded AAI matrix with 3 taxa.", "\n".join(captured.output))

    def test_load_matrix_accepts_fastaai_coded_numeric_values(self) -> None:
        """Accept FastAAI-coded matrix values, including 15.0 and 95.0."""
        with tempfile.TemporaryDirectory() as tempdir:
            matrix_path = Path(tempdir) / "FastAAI_matrix.txt"
            write_text(
                matrix_path,
                "\n".join(
                    [
                        "query_genome\tA\tB\tC",
                        "A\t95.0\t95.0\t15.0",
                        "B\t95.0\t95.0\t28.3",
                        "C\t15.0\t28.3\t95.0",
                    ]
                )
                + "\n",
            )

            names, ani, _name_to_idx = load_matrix(matrix_path)

            self.assertEqual(names, ["A", "B", "C"])
            self.assertAlmostEqual(ani[0, 1], 95.0)
            self.assertAlmostEqual(ani[0, 2], 15.0)
            self.assertAlmostEqual(ani[1, 2], 28.3)

    def test_load_matrix_rejects_duplicate_names(self) -> None:
        """Reject duplicate genome names in the header."""
        with tempfile.TemporaryDirectory() as tempdir:
            matrix_path = Path(tempdir) / "FastAAI_matrix.txt"
            write_text(
                matrix_path,
                "\n".join(
                    [
                        "query_genome\tA\tA",
                        "A\t95\t97",
                        "A\t97\t95",
                    ]
                )
                + "\n",
            )

            with self.assertRaises(SystemExit):
                load_matrix(matrix_path)

    def test_load_matrix_rejects_row_header_mismatch(self) -> None:
        """Reject matrices whose row and header names disagree."""
        with tempfile.TemporaryDirectory() as tempdir:
            matrix_path = Path(tempdir) / "FastAAI_matrix.txt"
            write_text(
                matrix_path,
                "\n".join(
                    [
                        "query_genome\tA\tB",
                        "A\t95\t97",
                        "C\t97\t95",
                    ]
                )
                + "\n",
            )

            with self.assertRaises(SystemExit):
                load_matrix(matrix_path)

    def test_load_matrix_rejects_non_numeric_values(self) -> None:
        """Reject matrices with non-numeric AAI values."""
        with tempfile.TemporaryDirectory() as tempdir:
            matrix_path = Path(tempdir) / "FastAAI_matrix.txt"
            write_text(
                matrix_path,
                "\n".join(
                    [
                        "query_genome\tA\tB",
                        "A\t95\tbad",
                        "B\t97\t95",
                    ]
                )
                + "\n",
            )

            with self.assertRaises(SystemExit):
                load_matrix(matrix_path)

    def test_load_matrix_rejects_asymmetric_values(self) -> None:
        """Reject matrices that are not symmetric."""
        with tempfile.TemporaryDirectory() as tempdir:
            matrix_path = Path(tempdir) / "FastAAI_matrix.txt"
            write_text(
                matrix_path,
                "\n".join(
                    [
                        "query_genome\tA\tB",
                        "A\t95\t97",
                        "B\t96.8\t95",
                    ]
                )
                + "\n",
            )

            with self.assertRaises(SystemExit):
                load_matrix(matrix_path)

    def test_load_and_check_tables_requires_lowercase_input_headers(self) -> None:
        """Reject input_list.tsv when it does not use lowercase headers."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            input_list_path = temp_path / "input_list.tsv"
            metadata_path = temp_path / "metadata.tsv"
            genome_path = temp_path / "A.fna"
            write_text(genome_path, ">A\nATGC\n")
            write_text(
                input_list_path,
                "Accession\tPath\n"
                f"A\t{genome_path}\n",
            )
            self.write_metadata_tsv(
                metadata_path,
                [self.metadata_row("A", "C1", "Organism_A")],
            )

            with self.assertRaises(SystemExit):
                load_and_check_tables(input_list_path, metadata_path, ["A"])

    def test_load_and_check_tables_accepts_lowercase_metadata_accession_header(self) -> None:
        """Accept lowercase accession in metadata.tsv."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            input_list_path = temp_path / "input_list.tsv"
            metadata_path = temp_path / "metadata.tsv"
            genome_path = temp_path / "A.fna"
            write_text(genome_path, ">A\nATGC\n")
            write_text(
                input_list_path,
                "accession\tpath\n"
                f"A\t{genome_path}\n",
            )
            self.write_metadata_tsv(
                metadata_path,
                [
                    self.metadata_row(
                        "A",
                        "C1",
                        "Organism_A",
                        accession_header="accession",
                    )
                ],
                accession_header="accession",
            )

            tsv, csv_by_acc, matrix_to_accession = load_and_check_tables(
                input_list_path,
                metadata_path,
                ["A"],
            )

            self.assertIn("A", tsv)
            self.assertIn("A", csv_by_acc)
            self.assertEqual(matrix_to_accession["A"], "A")

    def test_load_and_check_tables_accepts_composite_input_list_label_when_organism_name_is_na(self) -> None:
        """Resolve ${Cluster_ID}_${accession} in input_list.tsv back to the canonical accession."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            input_list_path = temp_path / "input_list.tsv"
            metadata_path = temp_path / "metadata.tsv"
            genome_path = temp_path / "A.fna"
            write_text(genome_path, ">A\nATGC\n")
            write_text(
                input_list_path,
                "accession\tpath\n"
                f"cluster.1_A\t{genome_path}\n",
            )
            self.write_metadata_tsv(
                metadata_path,
                [self.metadata_row("A", "cluster.1", "NA")],
            )

            with self.assertLogs(level="INFO") as captured:
                tsv, csv_by_acc, matrix_to_accession = load_and_check_tables(
                    input_list_path,
                    metadata_path,
                    ["cluster.1_A"],
                )

            self.assertIn("A", tsv)
            self.assertEqual(tsv["A"]["path"], str(genome_path))
            self.assertIn("A", csv_by_acc)
            self.assertEqual(matrix_to_accession["cluster.1_A"], "A")
            self.assertIn("input_list.tsv accession", "\n".join(captured.output))

    def test_load_and_check_tables_accepts_full_composite_input_list_label(self) -> None:
        """Resolve a full composite input-list label to the canonical metadata accession."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            input_list_path = temp_path / "input_list.tsv"
            metadata_path = temp_path / "metadata.tsv"
            genome_path = temp_path / "GCA_000018785.1.fna"
            matrix_label = "C000047_GCA_000018785_1_Acholeplasma_laidlawii_PG_8A"
            write_text(genome_path, ">A\nATGC\n")
            write_text(
                input_list_path,
                "accession\tpath\n"
                f"{matrix_label}\t{genome_path}\n",
            )
            self.write_metadata_tsv(
                metadata_path,
                [
                    self.metadata_row(
                        "GCA_000018785.1",
                        "C000047",
                        "Acholeplasma laidlawii PG-8A",
                    )
                ],
            )

            with self.assertLogs(level="INFO") as captured:
                tsv, csv_by_acc, matrix_to_accession = load_and_check_tables(
                    input_list_path,
                    metadata_path,
                    [matrix_label],
                )

            self.assertIn("GCA_000018785.1", tsv)
            self.assertEqual(tsv["GCA_000018785.1"]["path"], str(genome_path))
            self.assertIn("GCA_000018785.1", csv_by_acc)
            self.assertEqual(matrix_to_accession[matrix_label], "GCA_000018785.1")
            self.assertIn("input_list.tsv accession", "\n".join(captured.output))

    def test_load_and_check_tables_logs_info_for_raw_composite_match(self) -> None:
        """Log INFO when a matrix name matches the raw normalised composite alias."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            input_list_path = temp_path / "input_list.tsv"
            metadata_path = temp_path / "metadata.tsv"
            genome_path = temp_path / "A.fna"
            write_text(genome_path, ">A\nATGC\n")
            write_text(
                input_list_path,
                "accession\tpath\n"
                f"A\t{genome_path}\n",
            )
            self.write_metadata_tsv(
                metadata_path,
                [self.metadata_row("A", "cluster1", "Organism A")],
            )

            with self.assertLogs(level="INFO") as captured:
                _tsv, _csv_by_acc, matrix_to_accession = load_and_check_tables(
                    input_list_path,
                    metadata_path,
                    ["cluster1_A_Organism_A"],
                )

            self.assertEqual(matrix_to_accession["cluster1_A_Organism_A"], "A")
            self.assertIn("raw composite alias", "\n".join(captured.output))

    def test_load_and_check_tables_logs_info_for_sanitised_composite_match(self) -> None:
        """Log INFO when a matrix name matches the sanitised composite alias."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            input_list_path = temp_path / "input_list.tsv"
            metadata_path = temp_path / "metadata.tsv"
            genome_path = temp_path / "A.fna"
            write_text(genome_path, ">A\nATGC\n")
            write_text(
                input_list_path,
                "accession\tpath\n"
                f"A\t{genome_path}\n",
            )
            self.write_metadata_tsv(
                metadata_path,
                [self.metadata_row("A", "cluster.1", "Organism A")],
            )
            matrix_label = sanitise("cluster.1_A_Organism A")

            with self.assertLogs(level="INFO") as captured:
                _tsv, _csv_by_acc, matrix_to_accession = load_and_check_tables(
                    input_list_path,
                    metadata_path,
                    [matrix_label],
                )

            self.assertEqual(matrix_to_accession[matrix_label], "A")
            self.assertIn("sanitised composite alias", "\n".join(captured.output))

    def test_load_and_check_tables_sanitises_symbol_heavy_organism_names(self) -> None:
        """Replace non-allowed Organism_Name symbols with underscores for alias matching."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            input_list_path = temp_path / "input_list.tsv"
            metadata_path = temp_path / "metadata.tsv"
            genome_path = temp_path / "A.fna"
            write_text(genome_path, ">A\nATGC\n")
            write_text(
                input_list_path,
                "accession\tpath\n"
                f"A\t{genome_path}\n",
            )
            organism_name = "Organism (A)/B:C,+test"
            self.write_metadata_tsv(
                metadata_path,
                [self.metadata_row("A", "cluster.1", organism_name)],
            )
            matrix_label = sanitise(f"cluster.1_A_{organism_name}")

            with self.assertLogs(level="INFO") as captured:
                _tsv, _csv_by_acc, matrix_to_accession = load_and_check_tables(
                    input_list_path,
                    metadata_path,
                    [matrix_label],
                )

            self.assertEqual(matrix_to_accession[matrix_label], "A")
            self.assertIn("sanitised composite alias", "\n".join(captured.output))

    def test_normalise_organism_name_for_alias_replaces_all_separators(self) -> None:
        """Normalise every non-alphanumeric separator in Organism_Name to underscores."""
        self.assertEqual(
            normalise_organism_name_for_alias("Acholeplasma laidlawii PG-8A"),
            "Acholeplasma_laidlawii_PG_8A",
        )
        self.assertEqual(
            normalise_organism_name_for_alias("Organism (A)/B:C,+test"),
            "Organism_A_B_C_test",
        )
        self.assertIsNone(normalise_organism_name_for_alias("NA"))
        self.assertIsNone(normalise_organism_name_for_alias(""))

    def test_load_and_check_tables_matches_hyphenated_organism_names(self) -> None:
        """Match matrix labels that replace hyphens in Organism_Name with underscores."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            input_list_path = temp_path / "input_list.tsv"
            metadata_path = temp_path / "metadata.tsv"
            genome_path = temp_path / "GCA_000018785.1.fna"
            write_text(genome_path, ">A\nATGC\n")
            write_text(
                input_list_path,
                "accession\tpath\n"
                f"GCA_000018785.1\t{genome_path}\n",
            )
            self.write_metadata_tsv(
                metadata_path,
                [
                    self.metadata_row(
                        "GCA_000018785.1",
                        "C000047",
                        "Acholeplasma laidlawii PG-8A",
                    )
                ],
            )
            matrix_label = "C000047_GCA_000018785_1_Acholeplasma_laidlawii_PG_8A"

            with self.assertLogs(level="INFO") as captured:
                _tsv, _csv_by_acc, matrix_to_accession = load_and_check_tables(
                    input_list_path,
                    metadata_path,
                    [matrix_label],
                )

            self.assertEqual(matrix_to_accession[matrix_label], "GCA_000018785.1")
            self.assertIn("sanitised composite alias", "\n".join(captured.output))

    def test_load_and_check_tables_allows_cluster_accession_key_when_organism_name_is_na(self) -> None:
        """Use ${Cluster_ID}_${accession} when Organism_Name is NA."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            input_list_path = temp_path / "input_list.tsv"
            metadata_path = temp_path / "metadata.tsv"
            genome_path = temp_path / "A.fna"
            write_text(genome_path, ">A\nATGC\n")
            write_text(
                input_list_path,
                "accession\tpath\n"
                f"A\t{genome_path}\n",
            )
            self.write_metadata_tsv(
                metadata_path,
                [self.metadata_row("A", "cluster.1", "NA")],
            )
            matrix_label = "cluster.1_A"

            with self.assertLogs(level="INFO") as captured:
                _tsv, _csv_by_acc, matrix_to_accession = load_and_check_tables(
                    input_list_path,
                    metadata_path,
                    [matrix_label],
                )

            self.assertEqual(matrix_to_accession[matrix_label], "A")
            self.assertIn("raw composite alias", "\n".join(captured.output))

    def test_load_and_check_tables_allows_cluster_accession_key_when_organism_name_is_empty(self) -> None:
        """Use ${Cluster_ID}_${accession} when Organism_Name is empty."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            input_list_path = temp_path / "input_list.tsv"
            metadata_path = temp_path / "metadata.tsv"
            genome_path = temp_path / "A.fna"
            write_text(genome_path, ">A\nATGC\n")
            write_text(
                input_list_path,
                "accession\tpath\n"
                f"A\t{genome_path}\n",
            )
            self.write_metadata_tsv(
                metadata_path,
                [self.metadata_row("A", "cluster.1", "")],
            )
            matrix_label = sanitise("cluster.1_A")

            with self.assertLogs(level="INFO") as captured:
                _tsv, _csv_by_acc, matrix_to_accession = load_and_check_tables(
                    input_list_path,
                    metadata_path,
                    [matrix_label],
                )

            self.assertEqual(matrix_to_accession[matrix_label], "A")
            self.assertIn("sanitised composite alias", "\n".join(captured.output))

    def test_load_and_check_tables_rejects_conflicting_sanitised_aliases(self) -> None:
        """Reject metadata rows that collide after sanitisation."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            input_list_path = temp_path / "input_list.tsv"
            metadata_path = temp_path / "metadata.tsv"
            for accession in ("A.1", "A_1"):
                genome_path = temp_path / f"{accession}.fna"
                write_text(genome_path, f">{accession}\nATGC\n")
            write_text(
                input_list_path,
                "\n".join(
                    [
                        "accession\tpath",
                        f"A.1\t{temp_path / 'A.1.fna'}",
                        f"A_1\t{temp_path / 'A_1.fna'}",
                    ]
                )
                + "\n",
            )
            self.write_metadata_tsv(
                metadata_path,
                [
                    self.metadata_row("A.1", "cluster.1", "Organism A"),
                    self.metadata_row("A_1", "cluster_1", "Organism A"),
                ],
            )

            with self.assertRaises(SystemExit):
                load_and_check_tables(input_list_path, metadata_path, ["A"])

    def test_load_and_check_tables_rejects_duplicate_input_labels_resolving_to_same_accession(self) -> None:
        """Reject input-list labels that collapse onto one canonical metadata accession."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            input_list_path = temp_path / "input_list.tsv"
            metadata_path = temp_path / "metadata.tsv"
            genome_path_a = temp_path / "A1.fna"
            genome_path_b = temp_path / "A2.fna"
            write_text(genome_path_a, ">A\nATGC\n")
            write_text(genome_path_b, ">A\nATGC\n")
            write_text(
                input_list_path,
                "\n".join(
                    [
                        "accession\tpath",
                        f"A\t{genome_path_a}",
                        f"cluster1_A_Organism_A\t{genome_path_b}",
                    ]
                )
                + "\n",
            )
            self.write_metadata_tsv(
                metadata_path,
                [self.metadata_row("A", "cluster1", "Organism A")],
            )

            with self.assertRaises(SystemExit):
                load_and_check_tables(input_list_path, metadata_path, ["A"])

    def test_load_and_check_tables_rejects_unmatched_input_list_label(self) -> None:
        """Reject input-list accession labels that cannot be resolved through metadata."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            input_list_path = temp_path / "input_list.tsv"
            metadata_path = temp_path / "metadata.tsv"
            genome_path = temp_path / "A.fna"
            write_text(genome_path, ">A\nATGC\n")
            write_text(
                input_list_path,
                "accession\tpath\n"
                f"not_a_match\t{genome_path}\n",
            )
            self.write_metadata_tsv(
                metadata_path,
                [self.metadata_row("A", "cluster1", "Organism_A")],
            )

            with self.assertLogs(level="CRITICAL") as captured:
                with self.assertRaises(SystemExit):
                    load_and_check_tables(input_list_path, metadata_path, ["A"])

            self.assertIn(
                "Unmatched input_list.tsv accession labels",
                "\n".join(captured.output),
            )

    def test_load_and_check_tables_rejects_composite_match_without_cluster_id(self) -> None:
        """Reject composite fallback when Cluster_ID is missing for that metadata row."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            input_list_path = temp_path / "input_list.tsv"
            metadata_path = temp_path / "metadata.tsv"
            genome_path = temp_path / "A.fna"
            write_text(genome_path, ">A\nATGC\n")
            write_text(
                input_list_path,
                "accession\tpath\n"
                f"A\t{genome_path}\n",
            )
            self.write_metadata_tsv(
                metadata_path,
                [self.metadata_row("A", "", "Organism_A")],
            )

            with self.assertRaises(SystemExit):
                load_and_check_tables(
                    input_list_path,
                    metadata_path,
                    ["cluster1_A_Organism_A"],
                )

    def test_normalise_threshold_accepts_fraction_and_percent(self) -> None:
        """Accept both supported threshold forms."""
        for raw, expected in (
            ("0.45", 0.45),
            ("45", 0.45),
            ("0.65", 0.65),
            ("65", 0.65),
            ("0.9", 0.9),
            ("90", 0.9),
        ):
            with self.subTest(raw=raw):
                self.assertAlmostEqual(normalise_threshold(raw), expected)

    def test_taxonomic_level_recognises_strict_cutoffs(self) -> None:
        """Label the configured genus and family AAI thresholds."""
        self.assertEqual(taxonomic_level_for_threshold(0.65), "genus")
        self.assertEqual(taxonomic_level_for_threshold(0.45), "family")
        self.assertEqual(taxonomic_level_for_threshold(0.70), "custom")

    def test_normalise_threshold_rejects_invalid_values(self) -> None:
        """Reject unsupported threshold values."""
        for value in ("0", "0.95", "1", "95", "100", "101", "-5", "abc"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalise_threshold(value)

    def test_normalise_threshold_error_mentions_fastaai_matrix_cap(self) -> None:
        """Explain why thresholds above 90% are invalid for FastAAI matrices."""
        with self.assertRaises(ValueError) as context:
            normalise_threshold("95")
        message = str(context.exception)
        self.assertIn("FastAAI", message)
        self.assertIn(">90% AAI values to 95.0", message)

    def test_parse_args_defaults_to_strict_genus_clustering(self) -> None:
        """Default to the strict 65% complete-linkage genus cutoff."""
        argv = [
            str(MODULE_PATH),
            "--ani-matrix",
            "matrix.tsv",
            "--input-list",
            "input.tsv",
            "--metadata",
            "metadata.tsv",
            "--outdir",
            "out",
        ]
        with mock.patch("sys.argv", argv):
            args = parse_args()

        self.assertEqual(args.threshold, "0.65")
        self.assertEqual(args.linkage, "complete")

    def test_parse_args_rejects_unsupported_linkage(self) -> None:
        """Reject linkage methods that can introduce uncontrolled chaining."""
        argv = [
            str(MODULE_PATH),
            "--ani-matrix",
            "matrix.tsv",
            "--input-list",
            "input.tsv",
            "--metadata",
            "metadata.tsv",
            "--outdir",
            "out",
            "--linkage",
            "single",
        ]
        with mock.patch("sys.argv", argv), self.assertRaises(SystemExit):
            parse_args()

    def test_hierarchical_linkage_is_invariant_to_matrix_order(self) -> None:
        """Return the same linkage groups after permuting matrix rows."""
        import numpy as np

        names = ["A", "B", "C", "D"]
        aai = np.array(
            [
                [100.0, 95.0, 95.0, 50.0],
                [95.0, 100.0, 85.0, 50.0],
                [95.0, 85.0, 100.0, 50.0],
                [50.0, 50.0, 50.0, 100.0],
            ]
        )

        def accession_groups(
            matrix_names: list[str],
            matrix: "np.ndarray",
            linkage_method: str,
        ) -> set[frozenset[str]]:
            clusters = cluster_hierarchical(
                matrix,
                matrix_names,
                0.90,
                linkage_method=linkage_method,
            )
            return {
                frozenset(matrix_names[index] for index in members)
                for members in clusters.values()
            }

        order = [3, 2, 0, 1]
        permuted_names = [names[index] for index in order]
        permuted_aai = aai[np.ix_(order, order)]

        for linkage_method in ("average", "complete"):
            with self.subTest(linkage_method=linkage_method):
                expected = accession_groups(names, aai, linkage_method)
                observed = accession_groups(
                    permuted_names,
                    permuted_aai,
                    linkage_method,
                )
                self.assertEqual(observed, expected)

        self.assertEqual(
            accession_groups(names, aai, "average"),
            {frozenset({"A", "B", "C"}), frozenset({"D"})},
        )

    def test_average_and_complete_linkage_have_distinct_threshold_contracts(self) -> None:
        """Average may merge by mean AAI while complete enforces every pair."""
        import numpy as np

        names = ["A", "B", "C"]
        aai = np.array(
            [
                [100.0, 70.0, 70.0],
                [70.0, 100.0, 60.0],
                [70.0, 60.0, 100.0],
            ]
        )
        average = cluster_hierarchical(aai, names, 0.65, "average")
        complete = cluster_hierarchical(aai, names, 0.65, "complete")

        self.assertEqual(sorted(map(len, average.values())), [3])
        self.assertEqual(sorted(map(len, complete.values())), [1, 2])
        for members in complete.values():
            for left_position, left in enumerate(members):
                for right in members[left_position + 1 :]:
                    self.assertGreaterEqual(aai[left, right], 65.0)

    def test_hierarchical_linkage_rejects_unsupported_method(self) -> None:
        """Reject direct API requests for unsupported linkage methods."""
        import numpy as np

        with self.assertRaisesRegex(ValueError, "Unsupported linkage method"):
            cluster_hierarchical(
                np.array([[100.0, 95.0], [95.0, 100.0]]),
                ["A", "B"],
                0.90,
                "single",
            )

    def test_average_linkage_warns_that_threshold_is_not_strict(self) -> None:
        """Warn when exploratory linkage is used with a taxonomic cutoff."""
        import numpy as np

        aai = np.array(
            [
                [100.0, 70.0, 65.0],
                [70.0, 100.0, 60.0],
                [65.0, 60.0, 100.0],
            ]
        )
        with self.assertLogs(level="WARNING") as captured:
            cluster_hierarchical(aai, ["A", "B", "C"], 0.65, "average")

        warning = "\n".join(captured.output)
        self.assertIn("does not guarantee that every pair", warning)
        self.assertIn("strict genus-level clustering", warning)

    def test_assign_cluster_ids_uses_prefix_and_stable_order(self) -> None:
        """Prefix cluster IDs and order by size then accession."""
        results = [
            (1, "Z", 3, [5], []),
            (3, "B", 2, [1, 2, 3], []),
            (3, "A", 1, [0, 4, 6], []),
        ]

        rep_by_cid, idxs_by_cid = assign_cluster_ids(results, "grp")

        self.assertEqual(list(rep_by_cid.keys()), ["grp1", "grp2", "grp3"])
        self.assertEqual(rep_by_cid["grp1"], "A")
        self.assertEqual(idxs_by_cid["grp2"], [1, 2, 3])

    def test_select_representative_uses_small_c_weight_in_all_profiles(self) -> None:
        """Keep ANI centrality in the score, but at a much smaller weight for all profiles."""
        import numpy as np

        names = ["A", "B"]
        idxs = [0, 1]
        ani = np.array(
            [
                [100.0, 95.0],
                [95.0, 100.0],
            ],
            dtype=float,
        )
        meta = {
            "A": self.genome("A"),
            "B": self.genome("B", scaffolds=5),
        }

        for profile in ("default", "isolate", "mag"):
            with self.subTest(profile=profile):
                _size, rep_acc, dbg = select_representative_for_indices(
                    idxs,
                    names,
                    meta,
                    ani,
                    profile,
                )
                self.assertEqual(rep_acc, "B")
                self.assertIn("C=0.05", dbg[0])
                self.assertGreater(meta["A"].Score, 0.0)
                self.assertGreater(meta["B"].Score, meta["A"].Score)

    def test_select_representative_prefers_quality_over_centrality_with_small_c_weight(self) -> None:
        """Prefer better scaffold quality even when another genome is more central."""
        import numpy as np

        names = ["A", "B", "C"]
        idxs = [0, 1, 2]
        ani = np.array(
            [
                [100.0, 99.0, 99.0],
                [99.0, 100.0, 90.0],
                [99.0, 90.0, 100.0],
            ],
            dtype=float,
        )
        meta = {
            "A": self.genome("A", scaffolds=20),
            "B": self.genome("B", scaffolds=2),
            "C": self.genome("C", scaffolds=20),
        }

        size, rep_acc, dbg = select_representative_for_indices(
            idxs,
            names,
            meta,
            ani,
            "default",
        )

        self.assertEqual(size, 3)
        self.assertEqual(rep_acc, "B")
        self.assertIn("C=0.05", dbg[0])
        self.assertGreater(meta["A"].Score, meta["C"].Score)
        self.assertGreater(meta["B"].Score, meta["A"].Score)

    def test_average_linkage_prefers_threshold_compatible_representative(self) -> None:
        """Restrict average-linkage representatives to threshold-compatible centres."""
        import numpy as np

        names = ["A", "B", "C"]
        aai = np.array(
            [
                [100.0, 95.0, 95.0],
                [95.0, 100.0, 85.0],
                [95.0, 85.0, 100.0],
            ]
        )
        meta = {
            "A": self.genome(
                "A",
                assembly_level="Contig",
                assembly_rank=0,
                scaffolds=50,
            ),
            "B": self.genome(
                "B",
                assembly_level="Complete Genome",
                assembly_rank=3,
                scaffolds=1,
            ),
            "C": self.genome(
                "C",
                assembly_level="Contig",
                assembly_rank=0,
                scaffolds=50,
            ),
        }

        _size, representative, debug_lines = select_representative_for_indices(
            [0, 1, 2],
            names,
            meta,
            aai,
            "default",
            threshold=0.90,
            linkage_method="average",
        )

        self.assertEqual(representative, "A")
        self.assertIn("retained 1/3 candidate", "\n".join(debug_lines))

    def test_average_linkage_warns_and_falls_back_without_central_candidate(self) -> None:
        """Fall back to quality scoring when no member reaches every other member."""
        import numpy as np

        names = ["A", "B", "C"]
        aai = np.array(
            [
                [100.0, 95.0, 85.0],
                [95.0, 100.0, 85.0],
                [85.0, 85.0, 100.0],
            ]
        )
        meta = {
            "A": self.genome("A", assembly_level="Contig", assembly_rank=0),
            "B": self.genome("B", assembly_level="Complete Genome", assembly_rank=3),
            "C": self.genome("C", assembly_level="Contig", assembly_rank=0),
        }

        with self.assertLogs(level="WARNING") as captured:
            _size, representative, debug_lines = select_representative_for_indices(
                [0, 1, 2],
                names,
                meta,
                aai,
                "default",
                threshold=0.90,
                linkage_method="average",
            )

        self.assertEqual(representative, "B")
        self.assertIn("no representative candidate", "\n".join(captured.output))
        self.assertIn("fell back", "\n".join(debug_lines))

    def test_run_pipeline_writes_prefixed_cluster_outputs(self) -> None:
        """Run the full pipeline with a custom cluster prefix."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            matrix_path = temp_path / "FastAAI_matrix.txt"
            input_list_path = temp_path / "input_list.tsv"
            metadata_path = temp_path / "metadata.tsv"
            outdir = temp_path / "out"

            write_text(
                matrix_path,
                "\n".join(
                    [
                        "query_genome\tA\tB\tC\tD",
                        "A\t95\t97\t96\t70",
                        "B\t97\t95\t95.5\t70",
                        "C\t96\t95.5\t95\t70",
                        "D\t70\t70\t70\t95",
                    ]
                )
                + "\n",
            )

            genome_paths = {}
            for accession in ("A", "B", "C", "D"):
                genome_path = temp_path / f"{accession}.fna"
                write_text(genome_path, f">{accession}\nATGC\n")
                genome_paths[accession] = genome_path

            write_text(
                input_list_path,
                "\n".join(
                    [
                        "accession\tpath",
                        f"A\t{genome_paths['A']}",
                        f"B\t{genome_paths['B']}",
                        f"C\t{genome_paths['C']}",
                        f"D\t{genome_paths['D']}",
                    ]
                )
                + "\n",
            )
            self.write_metadata_tsv(
                metadata_path,
                [
                    {
                        "Accession": "A",
                        "Cluster_ID": "cluster1",
                        "Assembly_Name": "A_asm",
                        "Organism_Name": "Organism_A",
                        "Gcode": "11",
                        "N50": "100000",
                        "Assembly_Level": "Scaffold",
                        "BUSCO_bacillota_odb12": BUSCO,
                        "Scaffolds": "20",
                        "Genome_Size": "4000000",
                        "Completeness_gcode4": "97.0",
                        "Completeness_gcode11": "97.0",
                        "Contamination_gcode4": "1.0",
                        "Contamination_gcode11": "1.0",
                    },
                    {
                        "Accession": "B",
                        "Cluster_ID": "cluster1",
                        "Assembly_Name": "B_asm",
                        "Organism_Name": "Organism_B",
                        "Gcode": "11",
                        "N50": "250000",
                        "Assembly_Level": "Complete Genome",
                        "BUSCO_bacillota_odb12": BUSCO,
                        "Scaffolds": "5",
                        "Genome_Size": "4100000",
                        "Completeness_gcode4": "99.0",
                        "Completeness_gcode11": "99.0",
                        "Contamination_gcode4": "0.5",
                        "Contamination_gcode11": "0.5",
                    },
                    {
                        "Accession": "C",
                        "Cluster_ID": "cluster1",
                        "Assembly_Name": "C_asm",
                        "Organism_Name": "Organism_C",
                        "Gcode": "11",
                        "N50": "150000",
                        "Assembly_Level": "Chromosome",
                        "BUSCO_bacillota_odb12": BUSCO,
                        "Scaffolds": "8",
                        "Genome_Size": "4050000",
                        "Completeness_gcode4": "98.0",
                        "Completeness_gcode11": "98.0",
                        "Contamination_gcode4": "0.7",
                        "Contamination_gcode11": "0.7",
                    },
                    {
                        "Accession": "D",
                        "Cluster_ID": "cluster2",
                        "Assembly_Name": "D_asm",
                        "Organism_Name": "Organism_D",
                        "Gcode": "11",
                        "N50": "90000",
                        "Assembly_Level": "Contig",
                        "BUSCO_bacillota_odb12": BUSCO,
                        "Scaffolds": "40",
                        "Genome_Size": "3900000",
                        "Completeness_gcode4": "96.0",
                        "Completeness_gcode11": "96.0",
                        "Contamination_gcode4": "1.5",
                        "Contamination_gcode11": "1.5",
                    },
                ],
            )

            args = argparse.Namespace(
                ani_matrix=matrix_path,
                input_list=input_list_path,
                metadata=metadata_path,
                threshold="90",
                outdir=outdir,
                cluster_id_prefix="grp",
                threads=1,
                score_profile="default",
            )

            run_pipeline(args, threads=1)

            cluster_rows = outdir.joinpath("cluster.tsv").read_text(encoding="utf-8").splitlines()
            rep_rows = outdir.joinpath("representatives.tsv").read_text(encoding="utf-8").splitlines()

            self.assertEqual(cluster_rows[0], "Accession\tCluster_ID\tIs_Representative\tANI_to_Representative\tScore\tPath")
            self.assertEqual(rep_rows[0], "Cluster_ID\tRepresentative_Accession\tOrganism_Name\tCheckM2_Completeness\tCheckM2_Contamination\tBUSCO\tAssembly_Level\tN50\tCluster_Size")
            self.assertIn("A\tgrp1\tno", cluster_rows[1])
            self.assertIn("B\tgrp1\tyes", "\n".join(cluster_rows))
            self.assertIn("D\tgrp2\tyes", "\n".join(cluster_rows))
            self.assertIn("grp1\tB\tOrganism_B", "\n".join(rep_rows))
            self.assertIn("grp2\tD\tOrganism_D", "\n".join(rep_rows))

    def test_run_pipeline_rejects_thresholds_above_fastaai_limit(self) -> None:
        """Fail fast when the requested threshold is not meaningful for FastAAI output."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            matrix_path = temp_path / "FastAAI_matrix.txt"
            input_list_path = temp_path / "input_list.tsv"
            metadata_path = temp_path / "metadata.tsv"
            outdir = temp_path / "out"

            write_text(
                matrix_path,
                "\n".join(
                    [
                        "query_genome\tA\tB",
                        "A\t95.0\t95.0",
                        "B\t95.0\t95.0",
                    ]
                )
                + "\n",
            )

            for accession in ("A", "B"):
                genome_path = temp_path / f"{accession}.fna"
                write_text(genome_path, f">{accession}\nATGC\n")

            write_text(
                input_list_path,
                "\n".join(
                    [
                        "accession\tpath",
                        f"A\t{temp_path / 'A.fna'}",
                        f"B\t{temp_path / 'B.fna'}",
                    ]
                )
                + "\n",
            )
            self.write_metadata_tsv(
                metadata_path,
                [
                    {
                        "Accession": "A",
                        "Cluster_ID": "cluster1",
                        "Assembly_Name": "A_asm",
                        "Organism_Name": "Organism_A",
                        "Gcode": "11",
                        "N50": "100000",
                        "Assembly_Level": "Scaffold",
                        "BUSCO_bacillota_odb12": BUSCO,
                        "Scaffolds": "10",
                        "Genome_Size": "4000000",
                        "Completeness_gcode4": "98.0",
                        "Completeness_gcode11": "98.0",
                        "Contamination_gcode4": "0.5",
                        "Contamination_gcode11": "0.5",
                    },
                    {
                        "Accession": "B",
                        "Cluster_ID": "cluster1",
                        "Assembly_Name": "B_asm",
                        "Organism_Name": "Organism_B",
                        "Gcode": "11",
                        "N50": "100000",
                        "Assembly_Level": "Scaffold",
                        "BUSCO_bacillota_odb12": BUSCO,
                        "Scaffolds": "10",
                        "Genome_Size": "4000000",
                        "Completeness_gcode4": "98.0",
                        "Completeness_gcode11": "98.0",
                        "Contamination_gcode4": "0.5",
                        "Contamination_gcode11": "0.5",
                    },
                ],
            )

            args = argparse.Namespace(
                ani_matrix=matrix_path,
                input_list=input_list_path,
                metadata=metadata_path,
                threshold="95",
                outdir=outdir,
                cluster_id_prefix="cluster",
                threads=1,
                score_profile="default",
            )

            with self.assertRaises(SystemExit) as context:
                run_pipeline(args, threads=1)
            self.assertEqual(context.exception.code, 1)

    def test_run_pipeline_threshold_changes_cluster_membership(self) -> None:
        """A higher threshold should split a previously shared cluster."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            matrix_path = temp_path / "FastAAI_matrix.txt"
            input_list_path = temp_path / "input_list.tsv"
            metadata_path = temp_path / "metadata.tsv"
            outdir = temp_path / "out"

            write_text(
                matrix_path,
                "\n".join(
                    [
                        "query_genome\tA\tB\tC",
                        "A\t95.0\t95.0\t89.0",
                        "B\t95.0\t95.0\t89.0",
                        "C\t89.0\t89.0\t95.0",
                    ]
                )
                + "\n",
            )

            input_lines = ["accession\tpath"]
            metadata_rows = []
            for accession, n50, level in (
                ("A", "100000", "Scaffold"),
                ("B", "250000", "Complete Genome"),
                ("C", "150000", "Chromosome"),
            ):
                genome_path = temp_path / f"{accession}.fna"
                write_text(genome_path, f">{accession}\nATGC\n")
                input_lines.append(f"{accession}\t{genome_path}")
                metadata_rows.append(
                    {
                        "Accession": accession,
                        "Cluster_ID": "cluster1" if accession in ("A", "B") else "cluster2",
                        "Assembly_Name": f"{accession}_asm",
                        "Organism_Name": f"Organism_{accession}",
                        "Gcode": "11",
                        "N50": n50,
                        "Assembly_Level": level,
                        "BUSCO_bacillota_odb12": BUSCO,
                        "Scaffolds": "10",
                        "Genome_Size": "4000000",
                        "Completeness_gcode4": "98.0",
                        "Completeness_gcode11": "98.0",
                        "Contamination_gcode4": "0.5",
                        "Contamination_gcode11": "0.5",
                    }
                )

            write_text(input_list_path, "\n".join(input_lines) + "\n")
            self.write_metadata_tsv(metadata_path, metadata_rows)

            args = argparse.Namespace(
                ani_matrix=matrix_path,
                input_list=input_list_path,
                metadata=metadata_path,
                threshold="90",
                outdir=outdir,
                cluster_id_prefix="cluster",
                threads=1,
                score_profile="default",
            )

            run_pipeline(args, threads=1)
            cluster_text = outdir.joinpath("cluster.tsv").read_text(encoding="utf-8")

            self.assertIn("A\tcluster1", cluster_text)
            self.assertIn("B\tcluster1", cluster_text)
            self.assertIn("C\tcluster2", cluster_text)


if __name__ == "__main__":
    unittest.main()
