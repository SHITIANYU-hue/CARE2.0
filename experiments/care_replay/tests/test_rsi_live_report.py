import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_rsi_live_feedback import paired_summary


class PairedReportTest(unittest.TestCase):
    def test_constant_effect_has_exact_interval(self):
        result=paired_summary([[2.]*40]*3,10,1000)
        self.assertEqual(result['ci95'],[2.,2.])
        self.assertEqual(result['model_replicates'],3)
        self.assertEqual(result['paired_seed_wins'],120)

    def test_model_draw_variability_not_treated_as_120_independent_calls(self):
        result=paired_summary([[5.]*40,[-5.]*40,[0.]*40],11,10000)
        self.assertLess(result['ci95'][0],-2)
        self.assertGreater(result['ci95'][1],2)
        self.assertEqual(result['per_replicate_means'],[5.,-5.,0.])

    def test_single_model_draw_not_reported_as_replicated(self):
        with self.assertRaises(ValueError):paired_summary([[2.]*40],0,100)


if __name__=='__main__':unittest.main()
