import pytest

from fraude.data import generate_synthetic
from fraude.train import train


@pytest.fixture(scope="session")
def synthetic_df():
    return generate_synthetic(n=150_000, seed=7)


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory, synthetic_df):
    """Entrena una vez para toda la sesion de tests, en una carpeta temporal."""
    out = tmp_path_factory.mktemp("artifacts")
    train(synthetic_df, out, source="synthetic:test")
    return out
