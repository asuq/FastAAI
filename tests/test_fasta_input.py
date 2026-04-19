"""Tests for FASTA parsing and text-input decoding."""

import gzip
import tempfile
import unittest
from pathlib import Path

from fastaai.fastaai import agnostic_reader
from fastaai.fastaai import read_fasta


class FastaInputTests(unittest.TestCase):
	"""Verify FASTA parsing handles malformed and mixed-encoding inputs."""

	def test_read_fasta_rejects_sequence_before_header(self):
		"""Reject malformed FASTA files with sequence text before a header."""
		with tempfile.TemporaryDirectory() as tempdir:
			test_path = Path(tempdir) / "bad.faa"
			test_path.write_text("MPEPTIDE\n>seq1\nAAAA\n", encoding = "ascii")

			with self.assertRaises(ValueError) as context:
				read_fasta(str(test_path))

		self.assertIn("sequence data found before the first header", str(context.exception))

	def test_read_fasta_skips_blank_leading_lines(self):
		"""Allow blank lines before the first FASTA record."""
		with tempfile.TemporaryDirectory() as tempdir:
			test_path = Path(tempdir) / "good.faa"
			test_path.write_text("\n\n>seq1 sample\nAAAA\n", encoding = "ascii")

			contents, descriptions = read_fasta(str(test_path))

		self.assertEqual(contents["seq1"], "AAAA")
		self.assertEqual(descriptions["seq1"], "sample")

	def test_agnostic_reader_falls_back_to_latin_1(self):
		"""Decode non-UTF-8 text inputs with a Latin-1 fallback."""
		with tempfile.TemporaryDirectory() as tempdir:
			test_path = Path(tempdir) / "latin1.txt.gz"
			with gzip.open(test_path, "wb") as handle:
				handle.write(b"caf\xe9\n")

			reader = agnostic_reader(str(test_path))
			try:
				self.assertEqual(reader.read(), "caf" + chr(233) + "\n")
			finally:
				reader.close()


if __name__ == "__main__":
	unittest.main()
