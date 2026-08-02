"""Tests for the genome-to-protein preprocessing path."""

import tempfile
import unittest
from pathlib import Path

from fastaai.fastaai import new_pyrodigal_manager


class GenomeToProteinTests(unittest.TestCase):
	"""Verify predicted proteins remain available after export."""

	def test_run_for_fastaai_keeps_predicted_proteins_in_memory(self):
		"""Preserve in-memory proteins for the later HMM step."""
		example_genome = Path(__file__).resolve().parents[1] / "example_genomes"
		example_genome = example_genome / "Xanthomonas_albilineans_GCA_000962915_1.fna.gz"

		with tempfile.TemporaryDirectory() as tempdir:
			output_base = Path(tempdir) / "proteins.faa"
			manager = new_pyrodigal_manager(trans_tables=[11, 4], meta=False, verbose=False)
			manager.run_for_fastaai(
				genome_file=str(example_genome),
				compress=True,
				outaa=str(output_base),
			)

			total_proteins = sum(len(proteins) for proteins in manager.protein_seqs.values())
			self.assertGreater(total_proteins, 0)


if __name__ == "__main__":
	unittest.main()
