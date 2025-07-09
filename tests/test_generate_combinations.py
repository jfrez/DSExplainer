import unittest
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from DSExplainer import DSExplainer

class GenerateCombinationsTest(unittest.TestCase):
    def test_pairs_and_triplets_created(self):
        X = pd.DataFrame({
            'A': [1, 2],
            'B': [3, 4],
            'C': [5, 6]
        })
        explainer = object.__new__(DSExplainer)
        explainer.comb = 3
        explainer.scaler = MinMaxScaler()
        result = DSExplainer.generate_combinations(explainer, X, scaler=explainer.scaler)
        expected = ['A_x_B', 'A_x_C', 'B_x_C', 'A_x_B_x_C']
        for col in expected:
            self.assertIn(col, result.columns)

if __name__ == '__main__':
    unittest.main()
