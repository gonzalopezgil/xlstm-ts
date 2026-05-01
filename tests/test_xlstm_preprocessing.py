import unittest

import numpy as np

from ml.models.xlstm_ts.preprocessing import (
    create_feature_target_sequences,
    normalise_feature_target_split_data_xlstm,
    normalise_split_data_xlstm,
)


class XLSTMPreprocessingTest(unittest.TestCase):
    def test_create_feature_target_sequences_uses_features_for_x_and_raw_target_for_y(self):
        features = np.arange(260, dtype=float).reshape(-1, 1) + 10_000.0
        target = np.arange(260, dtype=float).reshape(-1, 1)
        dates = np.arange(260)

        x, y, sequence_dates = create_feature_target_sequences(features, target, dates)

        self.assertEqual(x[0, 0, 0].item(), 10_000.0)
        self.assertEqual(x[0, -1, 0].item(), 10_149.0)
        self.assertEqual(y[0, 0].item(), 150.0)
        self.assertEqual(sequence_dates.iloc[0], 150)

    def test_normalise_split_data_fits_scaler_on_training_values_only(self):
        train_x = np.array([[[0.0], [2.0]], [[4.0], [6.0]]])
        train_y = np.array([[8.0], [10.0]])
        val_x = np.array([[[1_000.0], [1_100.0]]])
        val_y = np.array([[1_200.0]])
        test_x = np.array([[[-500.0], [-400.0]]])
        test_y = np.array([[-300.0]])

        (
            train_x_scaled,
            train_y_scaled,
            val_x_scaled,
            val_y_scaled,
            test_x_scaled,
            test_y_scaled,
            scaler,
        ) = normalise_split_data_xlstm(
            train_x,
            train_y,
            val_x,
            val_y,
            test_x,
            test_y,
        )

        self.assertEqual(scaler.data_min_[0], 0.0)
        self.assertEqual(scaler.data_max_[0], 10.0)
        self.assertEqual(train_x_scaled.min(), 0.0)
        self.assertEqual(train_y_scaled.max(), 1.0)
        self.assertGreater(val_x_scaled.max(), 1.0)
        self.assertGreater(val_y_scaled.max(), 1.0)
        self.assertLess(test_x_scaled.min(), 0.0)
        self.assertLess(test_y_scaled.min(), 0.0)

    def test_feature_target_normalisation_uses_separate_train_only_scalers(self):
        train_x = np.array([[[100.0], [120.0]], [[140.0], [160.0]]])
        train_y = np.array([[0.0], [10.0]])
        val_x = np.array([[[1_000.0], [1_100.0]]])
        val_y = np.array([[1_200.0]])
        test_x = np.array([[[-500.0], [-400.0]]])
        test_y = np.array([[-300.0]])

        (
            train_x_scaled,
            train_y_scaled,
            val_x_scaled,
            val_y_scaled,
            test_x_scaled,
            test_y_scaled,
            x_scaler,
            y_scaler,
        ) = normalise_feature_target_split_data_xlstm(
            train_x,
            train_y,
            val_x,
            val_y,
            test_x,
            test_y,
        )

        self.assertEqual(x_scaler.data_min_[0], 100.0)
        self.assertEqual(x_scaler.data_max_[0], 160.0)
        self.assertEqual(y_scaler.data_min_[0], 0.0)
        self.assertEqual(y_scaler.data_max_[0], 10.0)
        self.assertEqual(train_x_scaled.min(), 0.0)
        self.assertEqual(train_y_scaled.max(), 1.0)
        self.assertGreater(val_x_scaled.max(), 1.0)
        self.assertGreater(val_y_scaled.max(), 1.0)
        self.assertLess(test_x_scaled.min(), 0.0)
        self.assertLess(test_y_scaled.min(), 0.0)


if __name__ == "__main__":
    unittest.main()
