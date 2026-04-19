"""Tests for HMM loading and agnostic file reading."""

import gzip
import tempfile
import unittest
from pathlib import Path

from fastaai.fastaai import agnostic_reader
from fastaai.fastaai import find_hmm
from fastaai.fastaai import new_pyhmmer_manager


class AgnosticReaderTests(unittest.TestCase):
	"""Verify plain-text and gzipped reads return text."""

	def test_read_plain_text(self):
		"""Read plain text as a string."""
		with tempfile.TemporaryDirectory() as tempdir:
			test_path = Path(tempdir) / "plain.txt"
			test_path.write_text("plain-text\n", encoding="ascii")
			reader = agnostic_reader(str(test_path))
			try:
				self.assertEqual(reader.read(), "plain-text\n")
			finally:
				reader.close()

	def test_read_gzipped_text(self):
		"""Read gzipped text as a decoded string."""
		with tempfile.TemporaryDirectory() as tempdir:
			test_path = Path(tempdir) / "plain.txt.gz"
			with gzip.open(test_path, "wb") as handle:
				handle.write(b"gz-text\n")
			reader = agnostic_reader(str(test_path))
			try:
				self.assertEqual(reader.read(), "gz-text\n")
			finally:
				reader.close()


class HmmLoaderTests(unittest.TestCase):
	"""Verify the bundled HMM file can be loaded."""

	def test_load_hmm_from_file(self):
		"""Load the bundled HMM models without a reader failure."""
		manager = new_pyhmmer_manager(compress=False)
		manager.load_hmm_from_file(find_hmm())
		self.assertGreater(len(manager.hmm_models), 0)


if __name__ == "__main__":
	unittest.main()
