from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def official_cache():
    cache = Path(__file__).resolve().parents[1] / ".cache/official_dico_nli"
    if not (cache / "evaluation_functions/__main__.py").is_file():
        pytest.skip("Fetch the pinned official scorer with make fetch-data for integration tests.")
    return cache
