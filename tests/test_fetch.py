import importlib.util
from io import BytesIO
from pathlib import Path

import pytest


def test_fetch_verifies_existing_bytes_and_requires_explicit_replacement(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/fetch_official_assets.py"
    spec = importlib.util.spec_from_file_location("fetch_official_assets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: BytesIO(b"pinned bytes"))
    destination = tmp_path / "asset.csv"
    record = module.download("https://example.test/pin/asset.csv", destination, force=False)
    assert len(record["sha256"]) == 64
    destination.write_bytes(b"different revision")
    with pytest.raises(RuntimeError, match="differs"):
        module.download(record["url"], destination, force=False)
    assert destination.read_bytes() == b"different revision"
    module.download(record["url"], destination, force=True)
    assert destination.read_bytes() == b"pinned bytes"
