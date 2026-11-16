# SPDX-License-Identifier: GPL-3.0-or-later
"""mouse_moe.download against a local HTTP server (no network, no PyTorch)."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from mouse_moe.config import RunConfig, load_config_dict
from mouse_moe.download import DownloadError, download, read_manifest, run_of, wanted_files

TRAIN = "training-run1-test80-20240401-0017.tar"
VALIDATE = "training-run1-test80-20240401-2317.tar"
GAME = "training-run3-test90-20251017-1817.tar"


class _Files(BaseHTTPRequestHandler):
    files: ClassVar[dict[str, bytes]] = {}
    honour_ranges: ClassVar[bool] = True
    requests: ClassVar[list[str]] = []

    def do_GET(self) -> None:  # noqa: N802  (the name http.server calls)
        name = self.path.rsplit("/", 1)[-1]
        type(self).requests.append(self.headers.get("Range", "full"))
        body = self.files.get(name)
        if body is None:
            self.send_error(404)
            return
        start = 0
        ranged = self.headers.get("Range")
        if ranged and self.honour_ranges:
            start = int(ranged.removeprefix("bytes=").rstrip("-"))
            self.send_response(206)
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        self.wfile.write(body[start:])

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    _Files.files = {
        TRAIN: b"t" * 50_000,
        VALIDATE: b"v" * 1_000,
        GAME: b"g" * 7_000,
    }
    _Files.honour_ranges = True
    _Files.requests = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Files)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/"
    httpd.shutdown()


def _manifests(tmp_path: Path, files: dict[str, bytes]) -> Path:
    root = tmp_path / "manifests"
    for name, body in files.items():
        run = run_of(name)
        manifest = root / run / "MANIFEST.csv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        if not manifest.exists():
            manifest.write_text('"file","bytes","sha256"\n', encoding="ascii")
        with manifest.open("a", encoding="ascii") as out:
            out.write(f'"{name}","{len(body)}","{hashlib.sha256(body).hexdigest()}"\n')
    return root


def _config() -> RunConfig:
    return load_config_dict(
        {
            "name": "download-test",
            "data": {
                "directory": "data/test80",
                "train_shards": [TRAIN],
                "validation_shards": [VALIDATE],
                "game_directory": "data/test90",
                "game_shards": [GAME],
            },
        }
    )


def _urls(base: str) -> dict[str, str]:
    return {"test80": base, "test90": base}


def test_downloads_verifies_and_writes_each_data_manifest(tmp_path: Path, server: str) -> None:
    manifests = _manifests(tmp_path, _Files.files)
    paths = download(_config(), tmp_path, manifests, _urls(server), report=lambda _: None,
                     disk_reserve=0)
    assert [p.name for p in paths] == [TRAIN, VALIDATE, GAME]
    assert (tmp_path / "data" / "test80" / TRAIN).read_bytes() == _Files.files[TRAIN]
    assert set(read_manifest(tmp_path / "data" / "test80" / "MANIFEST.csv")) == {TRAIN, VALIDATE}
    assert set(read_manifest(tmp_path / "data" / "test90" / "MANIFEST.csv")) == {GAME}
    assert not list(tmp_path.rglob("*.part"))


def test_a_present_matching_file_is_not_downloaded_again(tmp_path: Path, server: str) -> None:
    manifests = _manifests(tmp_path, _Files.files)
    download(_config(), tmp_path, manifests, _urls(server), report=lambda _: None, disk_reserve=0)
    _Files.requests.clear()
    download(_config(), tmp_path, manifests, _urls(server), report=lambda _: None, disk_reserve=0)
    assert _Files.requests == []


def test_a_partial_file_resumes_from_where_it_stopped(tmp_path: Path, server: str) -> None:
    manifests = _manifests(tmp_path, _Files.files)
    folder = tmp_path / "data" / "test80"
    folder.mkdir(parents=True)
    (folder / f"{TRAIN}.part").write_bytes(_Files.files[TRAIN][:12_345])
    download(_config(), tmp_path, manifests, _urls(server), report=lambda _: None, disk_reserve=0)
    assert "bytes=12345-" in _Files.requests
    assert (folder / TRAIN).read_bytes() == _Files.files[TRAIN]


def test_a_server_that_ignores_ranges_restarts_the_file(tmp_path: Path, server: str) -> None:
    _Files.honour_ranges = False
    manifests = _manifests(tmp_path, _Files.files)
    folder = tmp_path / "data" / "test80"
    folder.mkdir(parents=True)
    (folder / f"{TRAIN}.part").write_bytes(_Files.files[TRAIN][:12_345])
    download(_config(), tmp_path, manifests, _urls(server), report=lambda _: None, disk_reserve=0)
    assert (folder / TRAIN).read_bytes() == _Files.files[TRAIN]


def test_a_corrupt_download_is_deleted_and_reported(tmp_path: Path, server: str) -> None:
    manifests = _manifests(tmp_path, _Files.files)
    _Files.files = {**_Files.files, TRAIN: b"x" * 50_000}  # same size, different bytes
    with pytest.raises(DownloadError, match="do not match the manifest"):
        download(_config(), tmp_path, manifests, _urls(server), report=lambda _: None,
                 disk_reserve=0)
    folder = tmp_path / "data" / "test80"
    assert not (folder / TRAIN).exists()
    assert not (folder / f"{TRAIN}.part").exists()


def test_a_file_missing_from_the_manifests_is_refused(tmp_path: Path) -> None:
    manifests = _manifests(tmp_path, {VALIDATE: b"v", GAME: b"g"})
    with pytest.raises(DownloadError, match="is not in manifests/test80/MANIFEST.csv"):
        wanted_files(_config(), tmp_path, manifests)


def test_not_enough_disk_space_is_refused_before_downloading(tmp_path: Path, server: str) -> None:
    manifests = _manifests(tmp_path, _Files.files)
    with pytest.raises(DownloadError, match="GiB is free"):
        download(_config(), tmp_path, manifests, _urls(server), report=lambda _: None,
                 disk_reserve=1 << 62)
    assert _Files.requests == []


def test_every_shipped_config_names_only_files_in_the_manifests() -> None:
    from mouse_moe.config import load_config

    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "configs").rglob("*.toml")) + sorted((root / "examples").rglob("*.toml"))
    for path in paths:
        wanted_files(load_config(path), root, root / "manifests")
