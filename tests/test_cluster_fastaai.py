"""Tests for the FastAAI clustering helper."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "fastaai" / "cluster_fastaai.py"
MODULE_SPEC = importlib.util.spec_from_file_location("test_cluster_fastaai_module", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load test module from {MODULE_PATH}")
CLUSTER_FASTAAI = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = CLUSTER_FASTAAI
MODULE_SPEC.loader.exec_module(CLUSTER_FASTAAI)

assign_cluster_ids = CLUSTER_FASTAAI.assign_cluster_ids
load_matrix = CLUSTER_FASTAAI.load_matrix
load_and_check_tables = CLUSTER_FASTAAI.load_and_check_tables
normalise_threshold = CLUSTER_FASTAAI.normalise_threshold
run_pipeline = CLUSTER_FASTAAI.run_pipeline
sanitise = CLUSTER_FASTAAI.sanitise


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

    def test_load_and_check_tables_logs_info_for_raw_composite_match(self) -> None:
        """Log INFO when a matrix name matches the raw composite alias."""
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
                    ["cluster1_A_Organism A"],
                )

            self.assertEqual(matrix_to_accession["cluster1_A_Organism A"], "A")
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
        self.assertAlmostEqual(normalise_threshold("0.9"), 0.9)
        self.assertAlmostEqual(normalise_threshold("90"), 0.9)

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
