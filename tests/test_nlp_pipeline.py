import unittest
from drake_lyric_nlp import compute_mattr, welch_ttest

class TestDrakeNLP(unittest.TestCase):
    def test_mattr_short_sequence(self):
        tokens = ["flow", "rhythm", "beat", "flow"]
        self.assertAlmostEqual(compute_mattr(tokens, window=50), 3 / 4)

    def test_mattr_sliding_window(self):
        # 100 alternating tokens
        tokens = ["verse", "chorus"] * 50
        # In any 50-word window, exactly 2 unique words exist -> MATTR = 2/50 = 0.04
        self.assertAlmostEqual(compute_mattr(tokens, window=50), 0.04)

    def test_welch_ttest_significance(self):
        group_a = [-0.5, -0.6, -0.4, -0.7]
        group_b = [0.8, 0.9, 0.7, 0.85]
        t_stat, p_val, cohens_d = welch_ttest(group_a, group_b)
        self.assertLess(t_stat, -10.0)
        self.assertLess(p_val, 0.01)
        self.assertGreater(abs(cohens_d), 2.0)

if __name__ == "__main__":
    unittest.main()
