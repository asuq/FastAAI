"""Tests for single-query timing reporting."""

import datetime
import unittest

from fastaai.fastaai import input_file


class SingleQueryTimingTests(unittest.TestCase):
	"""Verify repeated timing printouts stay safe."""

	def test_partial_timings_is_idempotent(self):
		"""Allow partial timings to be reported more than once."""
		result = input_file("query.faa", write_outputs = False)
		start = datetime.datetime.now()
		result.init_time = start
		result.prot_pred_time = start + datetime.timedelta(seconds = 1)
		result.hmm_search_time = start + datetime.timedelta(seconds = 3)
		result.besthits_time = start + datetime.timedelta(seconds = 4)

		first = result.partial_timings()
		result.prot_pred_time, result.hmm_search_time, result.besthits_time = result.get_partial_timings()
		second = result.partial_timings()

		self.assertEqual(first, second)
		self.assertIn("Protein prediction: 1.0s", first)
		self.assertIn("HMM search: 2.0s", first)
		self.assertIn("Best hits: 1.0s", first)


if __name__ == "__main__":
	unittest.main()
