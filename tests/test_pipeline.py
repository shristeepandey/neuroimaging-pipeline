"""Unit tests for the neuroimaging pipeline."""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing import quality_control, preprocess_clinical_data
from analysis import descriptive_statistics, group_comparisons, correlation_analysis


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Create synthetic data for testing
        self.df = pd.DataFrame({
            'subject_id': [f'sub-{i:03d}' for i in range(1, 11)],
            'age': np.random.randint(18, 85, 10),
            'sex': np.random.choice(['M', 'F'], 10),
            'diagnosis': np.random.choice(['Control', 'MCI', 'Dementia'], 10, p=[0.4, 0.35, 0.25]),
            'education_years': np.random.randint(8, 20, 10),
            'mmse_score': np.random.randint(15, 30, 10),
            'brain_volume_cm3': np.random.normal(1100, 120, 10),
            'hippocampal_volume_cm3': np.random.normal(7.2, 0.8, 10),
            'cognitive_score': np.random.normal(50, 10, 10),
            'scan_quality': np.random.choice(['High', 'Medium', 'Low'], 10, p=[0.7, 0.25, 0.05])
        })

    def test_quality_control(self):
        Path("data/raw/oasis").mkdir(parents=True, exist_ok=True)
        self.df.to_csv("data/raw/oasis/participants.csv", index=False)
        qc = quality_control("data/raw/oasis/participants.csv")
        self.assertEqual(len(qc), 10)

    def test_preprocess_clinical_data(self):
        processed = preprocess_clinical_data(self.df)
        self.assertIn('sex_encoded', processed.columns)
        self.assertIn('age_zscore', processed.columns)


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'subject_id': [f'sub-{i:03d}' for i in range(1, 21)],
            'age': np.random.randint(18, 85, 20),
            'sex': np.random.choice(['M', 'F'], 20),
            'diagnosis': np.random.choice(['Control', 'Patient'], 20, p=[0.5, 0.5]),
            'education_years': np.random.randint(8, 20, 20),
            'mmse_score': np.random.randint(15, 30, 20),
            'brain_volume_cm3': np.random.normal(1100, 120, 20),
            'hippocampal_volume_cm3': np.random.normal(7.2, 0.8, 20),
            'cognitive_score': np.random.normal(50, 10, 20),
            'diagnosis_binary': np.random.randint(0, 2, 20)
        })

    def test_descriptive_statistics(self):
        stats = descriptive_statistics(self.df)
        self.assertIn('age_mean', stats)
        self.assertIn('cognitive_score_std', stats)

    def test_group_comparisons(self):
        results = group_comparisons(self.df)
        self.assertIn('brain_volume_cm3', results)
        self.assertIn('t_statistic', results['brain_volume_cm3'])

    def test_correlation_analysis(self):
        corr_matrix, sig_corrs = correlation_analysis(self.df)
        self.assertEqual(corr_matrix.shape[0], corr_matrix.shape[1])


if __name__ == '__main__':
    unittest.main()
