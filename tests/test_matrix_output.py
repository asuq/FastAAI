"""Tests for matrix output bookkeeping."""

import io
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from fastaai.fastaai import db_db_remake
from fastaai.fastaai import db_query
from fastaai.fastaai import generate_accessions_index


def _create_minimal_database(database_path: Path) -> None:
	"""Create a two-genome FastAAI database for query bookkeeping tests."""
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


class MatrixOutputTests(unittest.TestCase):
	"""Verify matrix partial files are validated before merging."""

	def test_count_result_groups_ignores_empty_worker_groups(self):
		"""Expect files only from worker groups that contain queries."""
		remake = db_db_remake()
		remake.query_gak = [
			(0, {0: object()}, [0]),
			(1, {}, []),
		]

		self.assertEqual(remake.count_result_groups(), 1)

	def test_validate_result_files_accepts_expected_unique_files(self):
		"""Accept the expected set of unique existing files."""
		remake = db_db_remake()
		with tempfile.TemporaryDirectory() as tempdir:
			paths = []
			for index in range(2):
				path = Path(tempdir) / f"part_{index}.txt"
				path.write_text("1\t2\n", encoding="ascii")
				paths.append(str(path))
			remake.num_result_groups = len(paths)
			remake.validate_result_files(paths)

	def test_validate_result_files_rejects_duplicate_paths(self):
		"""Reject duplicate file paths before matrix merge."""
		remake = db_db_remake()
		with tempfile.TemporaryDirectory() as tempdir:
			path = Path(tempdir) / "part.txt"
			path.write_text("1\t2\n", encoding="ascii")
			remake.num_result_groups = 2
			with self.assertRaises(RuntimeError):
				remake.validate_result_files([str(path), str(path)])

	def test_validate_result_files_rejects_missing_paths(self):
		"""Reject missing files before matrix merge."""
		remake = db_db_remake()
		with tempfile.TemporaryDirectory() as tempdir:
			path = Path(tempdir) / "part.txt"
			path.write_text("1\t2\n", encoding="ascii")
			missing = Path(tempdir) / "missing.txt"
			remake.num_result_groups = 2
			with self.assertRaises(RuntimeError):
				remake.validate_result_files([str(path), str(missing)])

	def test_matrix_query_handles_more_threads_than_genomes(self):
		"""Complete an on-disk matrix query when some worker groups are empty."""
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
					threads = 8,
					do_stdev = False,
					style = "matrix",
					in_mem = False,
					store_results = False,
				)

			matrix_path = query_output / "results" / "FastAAI_matrix.txt"
			matrix_rows = matrix_path.read_text(encoding = "ascii").splitlines()
			self.assertEqual(len(matrix_rows), 3)
			for row in matrix_rows:
				self.assertEqual(len(row.split("\t")), 3)


if __name__ == "__main__":
	unittest.main()
