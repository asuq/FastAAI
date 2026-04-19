"""Tests for the in-memory database-query path."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from fastaai.fastaai import build_db
from fastaai.fastaai import db_query
from fastaai.fastaai import flatten_cached_targets


class InMemoryDbQueryTests(unittest.TestCase):
	"""Verify cached in-memory target lists stay integer-typed."""

	def test_flatten_cached_targets_returns_int_array(self):
		"""Flatten cached target hits into one integer array."""
		target_hits = np.array(
			[
				np.array([1, 2], dtype = np.int32),
				np.array([3, 4], dtype = np.int32),
			],
			dtype = object,
		)
		selection = np.array([0, 1], dtype = np.int32)

		flattened = flatten_cached_targets(target_hits, selection)

		self.assertEqual(flattened.dtype, np.int32)
		np.testing.assert_array_equal(flattened, np.array([1, 2, 3, 4], dtype = np.int32))
		np.testing.assert_array_equal(np.bincount(flattened, minlength = 5), np.array([0, 1, 1, 1, 1]))

	def test_db_query_in_memory_completes_on_small_example_db(self):
		"""Run the in-memory query path on a two-genome example database."""
		example_genomes = Path(__file__).resolve().parents[1] / "example_genomes"
		selected_genomes = [
			example_genomes / "Xanthomonas_albilineans_GCA_000962915_1.fna.gz",
			example_genomes / "Xanthomonas_albilineans_GCA_000962925_1.fna.gz",
		]

		with tempfile.TemporaryDirectory() as tempdir:
			tempdir_path = Path(tempdir)
			input_dir = tempdir_path / "inputs"
			build_output = tempdir_path / "build"
			query_output = tempdir_path / "query"
			input_dir.mkdir()

			for genome_path in selected_genomes:
				(input_dir / genome_path.name).symlink_to(genome_path)

			with redirect_stdout(io.StringIO()):
				build_db(
					genomes = str(input_dir),
					proteins = None,
					hmms = None,
					db_name = "mini.db",
					output = str(build_output),
					threads = 2,
					verbose = False,
					do_compress = True,
				)

				db_query(
					query = str(build_output / "database" / "mini.db"),
					target = str(build_output / "database" / "mini.db"),
					verbose = False,
					output = str(query_output),
					threads = 4,
					do_stdev = False,
					style = "tsv",
					in_mem = True,
					store_results = False,
				)

			results_dir = query_output / "results"
			result_files = sorted(results_dir.glob("*_results.txt"))
			self.assertEqual(len(result_files), 2)
			for result_file in result_files:
				content = result_file.read_text(encoding = "ascii")
				self.assertIn("query\ttarget\tavg_jacc_sim", content)


if __name__ == "__main__":
	unittest.main()
