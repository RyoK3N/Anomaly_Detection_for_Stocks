import numpy as np
import pandas as pd

from services.preprocess import preprocess_data


def test_preprocess_sequences_shape():
    data = pd.DataFrame(
        {
            "Open": np.arange(70),
            "High": np.arange(70),
            "Low": np.arange(70),
            "Close": np.arange(70),
            "Volume": np.arange(70),
        }
    )
    sequences, scaler = preprocess_data(data, 10)
    assert sequences.shape == (60, 10, 8)
