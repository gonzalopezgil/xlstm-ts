import unittest

import numpy as np

from ml.data.preprocessing import wavelet_denoising


def assert_transform_is_causal(transform, values, cutoff):
    baseline = transform(values.copy())

    changed = values.copy()
    changed[cutoff + 1:] = changed[cutoff + 1:] + np.linspace(
        1_000.0,
        2_000.0,
        len(changed[cutoff + 1:]),
    )

    transformed = transform(changed)

    np.testing.assert_allclose(
        baseline[: cutoff + 1],
        transformed[: cutoff + 1],
        rtol=1e-10,
        atol=1e-10,
        err_msg="Future observations changed denoised historical values.",
    )


class WaveletDenoisingCausalityTest(unittest.TestCase):
    def test_wavelet_denoising_does_not_use_future_observations(self):
        rng = np.random.default_rng(7)
        values = np.sin(np.linspace(0, 8 * np.pi, 256)) + rng.normal(0, 0.05, 256)

        assert_transform_is_causal(wavelet_denoising, values, cutoff=127)


if __name__ == "__main__":
    unittest.main()
