import unittest

from safetensors_to_litertlm.utils import behavior_benchmark


class BehaviorBenchmarkTests(unittest.TestCase):
    def test_score_response_counts_refusal_and_style_markers(self) -> None:
        text = "I'm sorry, as an AI I cannot assist with harmful requests."
        scores = behavior_benchmark.score_response(text)
        self.assertGreaterEqual(scores["refusal_hits"], 2)
        self.assertGreaterEqual(scores["style_hits"], 2)

    def test_aggregate_scores(self) -> None:
        summary = behavior_benchmark.aggregate_scores(
            [
                {"scores": {"refusal_hits": 2, "style_hits": 1}},
                {"scores": {"refusal_hits": 0, "style_hits": 3}},
            ]
        )
        self.assertEqual(summary["avg_refusal_hits"], 1.0)
        self.assertEqual(summary["avg_style_hits"], 2.0)


if __name__ == "__main__":
    unittest.main()
