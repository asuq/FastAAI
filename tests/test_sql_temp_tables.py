"""Tests for temporary SQLite table naming."""

import sqlite3
import unittest

from fastaai.fastaai import build_temp_table_name


class SqlTempTableTests(unittest.TestCase):
	"""Verify generated temporary table names stay SQLite-safe."""

	def test_build_temp_table_name_handles_digit_prefixed_query_names(self):
		"""Generate temp table names that SQLite can create directly."""
		temp_name = build_temp_table_name("11_0_1__xyz", "PF01813.17")
		self.assertTrue(temp_name.startswith("tmp_n_11_0_1_xyz_PF01813_17"))

		conn = sqlite3.connect(":memory:")
		try:
			conn.execute("CREATE TEMP TABLE " + temp_name + " (kmer INTEGER)")
		finally:
			conn.close()


if __name__ == "__main__":
	unittest.main()
