import unittest

from virel_budget.metrics import exact_match, reliance_retention, summarize_decisions, visual_evidence_margin
from virel_budget.schema import PolicyDecision, canonicalize_answer


class MetricsTest(unittest.TestCase):
    def test_exact_match_normalizes_simple_answers(self):
        self.assertTrue(exact_match("Yes.", "yes"))
        self.assertTrue(exact_match(" TRUE ", "yes"))
        self.assertFalse(exact_match("red", "blue"))

    def test_vem_and_reliance_retention(self):
        vem = visual_evidence_margin(-0.2, -1.1)
        self.assertAlmostEqual(vem, 0.9)
        self.assertAlmostEqual(reliance_retention(vem, 1.8), 0.5)
        self.assertIsNone(reliance_retention(vem, 0.0))

    def test_api_dense_budget_is_not_reported_as_dense_avoidance(self):
        decision = PolicyDecision(
            sample_id="a",
            split="test",
            dataset="d",
            gold_answer="green",
            method="dense_only",
            selected_method="dense",
            selected_budget="api",
            accepted=True,
            escalations=0,
            answer="green",
            is_correct=True,
            vem=1.0,
            dense_vem=1.0,
            reliance_retention=1.0,
            cost=0.01,
            latency_ms=100.0,
            token_count=80,
            reason="fixed_budget",
        )
        self.assertEqual(summarize_decisions([decision])["dense_avoidance_rate"], 0.0)

    def test_canonicalize_answer_handles_short_vlm_phrases(self):
        self.assertEqual(canonicalize_answer("The answer is yes.", options=["yes", "no"]), "yes")
        self.assertEqual(canonicalize_answer("Answer: A", option_map={"A": "red square"}), "red square")
        self.assertEqual(canonicalize_answer("B", option_map={"A": "red square", "B": "blue circle"}), "blue circle")
        self.assertEqual(canonicalize_answer("It appears to be a red square.", options=["red square", "blue circle"]), "red square")


if __name__ == "__main__":
    unittest.main()
