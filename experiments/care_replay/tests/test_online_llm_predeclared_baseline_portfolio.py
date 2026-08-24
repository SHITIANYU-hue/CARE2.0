import importlib.util
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "predeclared_portfolio",
    SCRIPTS / "build_online_llm_predeclared_baseline_portfolio.py",
)
portfolio = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(portfolio)


class PredeclaredBaselinePortfolioTests(unittest.TestCase):
    def test_exact_sign_test_ignores_ties(self):
        result = portfolio.exact_sign_test([1.0, 2.0, -1.0, 0.0])
        self.assertEqual(result["wins"], 2)
        self.assertEqual(result["losses"], 1)
        self.assertEqual(result["ties"], 1)
        self.assertEqual(result["two_sided_p"], 1.0)

    def test_all_positive_sign_test(self):
        result = portfolio.exact_sign_test([1.0] * 6)
        self.assertTrue(math.isclose(result["two_sided_p"], 0.03125))

    def test_bootstrap_is_reproducible(self):
        first = portfolio.bootstrap_mean_ci([1.0, 2.0, 3.0], samples=500, seed=9)
        second = portfolio.bootstrap_mean_ci([1.0, 2.0, 3.0], samples=500, seed=9)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], 2.0)
        self.assertGreaterEqual(first[1], 2.0)

    def test_extended_config_inherits_base_cases(self):
        config = portfolio.load_portfolio_config(
            ROOT / "configs" / "online_llm_predeclared_baseline_portfolio_v2.json"
        )
        self.assertEqual(len(config["cases"]), 17)
        self.assertEqual(
            config["protocol"]["version"],
            "care2_online_llm_complete_retrospective_portfolio_v2",
        )


if __name__ == "__main__":
    unittest.main()
