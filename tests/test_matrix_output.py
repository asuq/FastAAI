"""Tests for matrix output bookkeeping."""

import tempfile
import unittest
from pathlib import Path

from fastaai.fastaai import db_db_remake


class MatrixOutputTests(unittest.TestCase):
	"""Verify matrix partial files are validated before merging."""

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


if __name__ == "__main__":
	unittest.main()
