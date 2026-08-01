import unittest

from virel_budget.policies import Thresholds, apply_online_cascade_policy, apply_policy
from virel_budget.schema import EvalRecord


def record(sample_id, budget, intervention, vem, rr, cost=1.0):
    return EvalRecord(
        sample_id=sample_id,
        split="test",
        dataset="d",
        method="random",
        budget=budget,
        intervention=intervention,
        answer="yes",
        gold_answer="yes",
        is_correct=True,
        logprob_original=-0.1,
        logprob_intervened=-0.5,
        confidence=0.9,
        vem=vem,
        dense_vem=1.0,
        reliance_retention=rr,
        delta_vem=1.0 - vem,
        shortcut_persistence=False,
        latency_ms=10.0,
        token_count=16 if budget != "full" else 64,
        cost=cost,
        energy_joule=0.1,
    )


class PolicyTest(unittest.TestCase):
    def test_budget_must_pass_all_interventions(self):
        records = [
            record("a", 16, "gray", 1.0, 0.8),
            record("a", 16, "blur", 0.4, 0.8),
            record("a", 32, "gray", 1.0, 0.8),
            record("a", 32, "blur", 0.9, 0.8),
        ]
        decisions = apply_policy(records, Thresholds(tau=0.75, rho=0.7, dense_vem_min=0.1))
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].selected_budget, 32)
        self.assertEqual(decisions[0].escalations, 1)
        self.assertTrue(decisions[0].accepted)

    def test_online_policy_counts_failed_probe_cost(self):
        records = [
            record("a", 16, "gray", 0.1, 0.1, cost=2.0),
            record("a", 32, "gray", 1.0, 0.8, cost=3.0),
        ]
        decisions = apply_online_cascade_policy(records, Thresholds(tau=0.75, rho=0.7, dense_vem_min=0.1))
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].selected_budget, 32)
        self.assertAlmostEqual(decisions[0].cost, 5.0)
        self.assertAlmostEqual(decisions[0].online_cumulative_cost, 5.0)


if __name__ == "__main__":
    unittest.main()
