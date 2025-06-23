import hypothesis.strategies as st
import pandas as pd
from hypothesis import given

from services.technical_indicators import add_indicators


@given(st.lists(st.floats(1, 100), min_size=60, max_size=60))
def test_add_indicators(values):
    df = pd.DataFrame({"Close": values})
    result = add_indicators(df)
    assert set(["MA20", "MA50", "RSI"]).issubset(result.columns)
    assert len(result) == len(df)
    assert not result[["MA20", "MA50", "RSI"]].isna().any().any()
