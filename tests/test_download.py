"""Tests for the friendly authentication errors in utils/download.py."""

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

import muscriptor.utils.download as dl


def test_http_download_progress_does_not_pollute_stdout(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(dl, "_CACHE_DIR", tmp_path)

    def fake_retrieve(_url, destination):
        destination.write_bytes(b"weights")

    monkeypatch.setattr(dl.urllib.request, "urlretrieve", fake_retrieve)

    dl.download_if_necessary("https://example.com/model.safetensors")
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "Downloading model.safetensors" in captured.err


@pytest.mark.parametrize("exc", [GatedRepoError, RepositoryNotFoundError])
def test_gated_repo_maps_to_model_download_error(monkeypatch, exc):
    response = httpx.Response(
        status_code=401,
        request=httpx.Request("GET", "https://huggingface.co/api/models/x"),
    )

    def fake_download(repo_id, filename):
        raise exc("401 Client Error", response=response)

    monkeypatch.setattr(dl, "hf_hub_download", fake_download)
    with pytest.raises(dl.ModelDownloadError) as e:
        dl.download_if_necessary("hf://MuScriptor/muscriptor-medium/model.safetensors")
    msg = str(e.value)
    assert "hf auth login" in msg
    assert "HF_TOKEN" in msg
    assert "https://huggingface.co/MuScriptor/muscriptor-medium" in msg
