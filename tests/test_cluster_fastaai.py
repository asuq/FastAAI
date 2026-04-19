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
normalise_threshold = CLUSTER_FASTAAI.normalise_threshold
run_pipeline = CLUSTER_FASTAAI.run_pipeline


BUSCO = "C:98.0%[S:97.0%,D:1.0%],F:1.0%,M:1.0%,n:200"


def write_text(path: Path, content: str) -> None:
    """Write ASCII test content to a file."""
    path.write_text(content, encoding="ascii")


class ClusterFastAAITests(unittest.TestCase):
    """Verify FastAAI matrix parsing and clustering behaviour."""

    def test_load_matrix_accepts_valid_square_matrix(self) -> None:
        """Load a valid FastAAI matrix and normalise the diagonal."""
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

            names, ani, name_to_idx = load_matrix(matrix_path)

            self.assertEqual(names, ["A", "B", "C"])
            self.assertEqual(name_to_idx["B"], 1)
            self.assertAlmostEqual(ani[0, 1], 97.0)
            self.assertAlmostEqual(ani[1, 2], 95.5)
            self.assertAlmostEqual(ani[0, 0], 100.0)

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
        """Reject matrices with non-numeric ANI values."""
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
            metadata_path = temp_path / "metadata.csv"
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
                        "Accession\tAssembly_Name\tOrganism_Name\tPath",
                        f"A\tA_asm\tOrganism_A\t{genome_paths['A']}",
                        f"B\tB_asm\tOrganism_B\t{genome_paths['B']}",
                        f"C\tC_asm\tOrganism_C\t{genome_paths['C']}",
                        f"D\tD_asm\tOrganism_D\t{genome_paths['D']}",
                    ]
                )
                + "\n",
            )

            with metadata_path.open("w", encoding="ascii", newline="") as handle:
                fieldnames = [
                    "Accession",
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
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                rows = [
                    {
                        "Accession": "A",
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
                ]
                writer.writerows(rows)

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
            metadata_path = temp_path / "metadata.csv"
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
                        "Accession\tAssembly_Name\tOrganism_Name\tPath",
                        f"A\tA_asm\tOrganism_A\t{temp_path / 'A.fna'}",
                        f"B\tB_asm\tOrganism_B\t{temp_path / 'B.fna'}",
                    ]
                )
                + "\n",
            )

            with metadata_path.open("w", encoding="ascii", newline="") as handle:
                fieldnames = [
                    "Accession",
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
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for accession in ("A", "B"):
                    writer.writerow(
                        {
                            "Accession": accession,
                            "Assembly_Name": f"{accession}_asm",
                            "Organism_Name": f"Organism_{accession}",
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
            metadata_path = temp_path / "metadata.csv"
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

            input_lines = ["Accession\tAssembly_Name\tOrganism_Name\tPath"]
            metadata_rows = []
            for accession, n50, level in (
                ("A", "100000", "Scaffold"),
                ("B", "250000", "Complete Genome"),
                ("C", "150000", "Chromosome"),
            ):
                genome_path = temp_path / f"{accession}.fna"
                write_text(genome_path, f">{accession}\nATGC\n")
                input_lines.append(
                    f"{accession}\t{accession}_asm\tOrganism_{accession}\t{genome_path}"
                )
                metadata_rows.append(
                    {
                        "Accession": accession,
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

            with metadata_path.open("w", encoding="ascii", newline="") as handle:
                fieldnames = list(metadata_rows[0].keys())
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metadata_rows)

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
