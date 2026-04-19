"""Tests for the in-memory database-query path."""

import io
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from fastaai.fastaai import db_query
from fastaai.fastaai import flatten_cached_targets
from fastaai.fastaai import generate_accessions_index


def _create_minimal_database(database_path: Path) -> None:
	"""Create a two-genome FastAAI database for query tests."""
	accession = "PF01780_19"
	accession_id = generate_accessions_index()[accession]
	genome_kmers = {
		0: np.array([1, 2], dtype = np.int32),
		1: np.array([1, 3], dtype = np.int32),
	}
	kmer_genomes = {
		1: np.array([0, 1], dtype = np.int32),
		2: np.array([0], dtype = np.int32),
		3: np.array([1], dtype = np.int32),
	}

	with sqlite3.connect(database_path) as connection:
		connection.execute("CREATE TABLE genome_index (genome text, gen_id integer, protein_count integer)")
		connection.execute("CREATE TABLE genome_acc_kmer_counts (genome integer, accession integer, count integer)")
		connection.execute(f"CREATE TABLE {accession}_genomes (genome INTEGER PRIMARY KEY, kmers array)")
		connection.execute(f"CREATE TABLE {accession} (kmer INTEGER PRIMARY KEY, genomes array)")
		connection.executemany(
			"INSERT INTO genome_index VALUES (?, ?, ?)",
			[("genome_0", 0, 1), ("genome_1", 1, 1)],
		)
		connection.executemany(
			"INSERT INTO genome_acc_kmer_counts VALUES (?, ?, ?)",
			[(genome, accession_id, len(kmers)) for genome, kmers in genome_kmers.items()],
		)
		connection.executemany(
			f"INSERT INTO {accession}_genomes VALUES (?, ?)",
			[(genome, kmers.tobytes()) for genome, kmers in genome_kmers.items()],
		)
		connection.executemany(
			f"INSERT INTO {accession} VALUES (?, ?)",
			[(kmer, genomes.tobytes()) for kmer, genomes in kmer_genomes.items()],
		)


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
		with tempfile.TemporaryDirectory() as tempdir:
			tempdir_path = Path(tempdir)
			database_path = tempdir_path / "mini.db"
			query_output = tempdir_path / "query"
			_create_minimal_database(database_path)

			with redirect_stdout(io.StringIO()):
				db_query(
					query = str(database_path),
					target = str(database_path),
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
