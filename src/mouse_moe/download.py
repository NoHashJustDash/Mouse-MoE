# SPDX-License-Identifier: GPL-3.0-or-later
"""Download the Lc0 training files a config names, and check each one against the manifests.

    python -I -m mouse_moe.download --config examples\\default.toml

The files come from the Lc0 project's public servers. Each is written to the config's data
folder as ``<name>.part`` and renamed only once its size and SHA-256 match
``manifests/<run>/MANIFEST.csv``. A stopped download resumes where it left off. A file that
is already present and matches is skipped. Each data folder gets a ``MANIFEST.csv`` listing
the files in it, which the training loader requires.

The data is under the Open Database License; see docs/3-data.md.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from mouse_moe.config import RunConfig, load_config

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "manifests"
# Where each Lc0 run's files are served (checked 2026-11-16). The test90 folder on
# storage.lczero.org lists only empty archives; data.lczero.org holds the data.
BASE_URLS: Mapping[str, str] = {
    "test80": "https://storage.lczero.org/files/training_data/test80/",
    "test90": "https://data.lczero.org/files/training_data/test90/",
}
_RUN = re.compile(r"^training-run\d+-(test\d+)-\d{8}-\d{4}\.tar$")
CHUNK = 1 << 22
USER_AGENT = "mouse-moe-download"
# Free disk space left after every download, so the machine stays usable.
DISK_RESERVE = 10 * 1024**3


class DownloadError(RuntimeError):
    """A file could not be downloaded, or does not match its manifest."""


@dataclass(frozen=True)
class Wanted:
    name: str
    run: str
    directory: Path
    size: int
    sha256: str


def read_manifest(path: Path) -> dict[str, tuple[int, str]]:
    """``file,bytes,sha256`` rows, quoted, after a header line."""
    if not path.is_file():
        raise DownloadError(f"missing manifest {path}")
    entries: dict[str, tuple[int, str]] = {}
    for line in path.read_text(encoding="ascii").splitlines()[1:]:
        name, size, digest = (part.strip('"') for part in line.split(","))
        entries[name] = (int(size), digest)
    return entries


def write_manifest(path: Path, entries: Mapping[str, tuple[int, str]]) -> None:
    rows = ['"file","bytes","sha256"']
    rows += [f'"{name}","{size}","{digest}"' for name, (size, digest) in sorted(entries.items())]
    temporary = path.with_suffix(".csv.tmp")
    temporary.write_text("\n".join(rows) + "\n", encoding="ascii")
    temporary.replace(path)


def run_of(name: str) -> str:
    match = _RUN.match(name)
    if match is None:
        raise DownloadError(f"{name} is not an Lc0 training file name (training-runN-testNN-...)")
    return match.group(1)


def wanted_files(config: RunConfig, root: Path, manifests: Path) -> list[Wanted]:
    """Every file the config trains, validates or tests on, with its expected size and hash."""
    d = config.data
    groups = [(d.directory, [*d.train_shards, *d.validation_shards, *d.test_shards])]
    if d.game_shards:
        if not d.game_directory:
            raise DownloadError("the config lists game_shards but no game_directory")
        groups.append((d.game_directory, list(d.game_shards)))
    wanted: list[Wanted] = []
    seen: set[Path] = set()
    loaded: dict[str, dict[str, tuple[int, str]]] = {}
    for directory, names in groups:
        for name in names:
            path = (root / directory / name).resolve()
            if path in seen:
                continue
            seen.add(path)
            run = run_of(name)
            if run not in BASE_URLS:
                raise DownloadError(f"no download address for run {run} ({name})")
            if run not in loaded:
                loaded[run] = read_manifest(manifests / run / "MANIFEST.csv")
            if name not in loaded[run]:
                raise DownloadError(
                    f"{name} is not in manifests/{run}/MANIFEST.csv; add its size and SHA-256"
                )
            size, digest = loaded[run][name]
            wanted.append(Wanted(name, run, path.parent, size, digest))
    return wanted


def _hash_prefix(path: Path) -> hashlib._Hash:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
    return digest


def fetch(
    item: Wanted,
    url: str,
    report: Callable[[str], None] = print,
    opener: Callable[[urllib.request.Request], object] = urllib.request.urlopen,
) -> Path:
    """Download one file to its directory, resuming a ``.part`` file, and verify it."""
    target = item.directory / item.name
    if target.is_file() and target.stat().st_size == item.size:
        if _hash_prefix(target).hexdigest() == item.sha256:
            report(f"ok       {item.name} (already present)")
            return target
        raise DownloadError(f"{target} has the right size but the wrong SHA-256; delete it")
    item.directory.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if have > item.size:
        part.unlink()
        have = 0
    digest = _hash_prefix(part) if have else hashlib.sha256()
    if have < item.size:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        if have:
            request.add_header("Range", f"bytes={have}-")
        with opener(request) as response:  # type: ignore[attr-defined]
            status = response.status
            if have and status != 206:
                # The server ignored the range: start the file again.
                have, digest = 0, hashlib.sha256()
            elif status not in (200, 206):
                raise DownloadError(f"{url}: HTTP {status}")
            with part.open("ab" if have else "wb") as out:
                while block := response.read(CHUNK):
                    out.write(block)
                    digest.update(block)
                    have += len(block)
                    if have > item.size:
                        break
    if have != item.size or digest.hexdigest() != item.sha256:
        part.unlink(missing_ok=True)
        raise DownloadError(
            f"{item.name}: got {have} bytes that do not match the manifest "
            f"({item.size} bytes, SHA-256 {item.sha256}); the partial file was deleted"
        )
    part.replace(target)
    report(f"verified {item.name} ({item.size:,} bytes)")
    return target


def download(
    config: RunConfig,
    root: Path = ROOT,
    manifests: Path = MANIFESTS,
    base_urls: Mapping[str, str] = BASE_URLS,
    report: Callable[[str], None] = print,
    disk_reserve: int = DISK_RESERVE,
) -> list[Path]:
    """Download and verify every file ``config`` needs; return their paths."""
    wanted = wanted_files(config, root, manifests)
    missing = 0
    for item in wanted:
        target = item.directory / item.name
        part = target.with_name(target.name + ".part")
        if not target.is_file():
            missing += item.size - (part.stat().st_size if part.is_file() else 0)
    if missing:
        probe = next(item.directory for item in wanted)
        probe.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(probe).free
        if free - missing < disk_reserve:
            raise DownloadError(
                f"needs {missing / 1024**3:.1f} GiB but only {free / 1024**3:.1f} GiB is free "
                f"(keeping {disk_reserve / 1024**3:.0f} GiB spare)"
            )
    report(f"{len(wanted)} files, {sum(i.size for i in wanted) / 1024**3:.1f} GiB in total, "
           f"{missing / 1024**3:.1f} GiB to download")
    paths = [fetch(item, base_urls[item.run] + item.name, report) for item in wanted]
    for directory in sorted({item.directory for item in wanted}):
        manifest = directory / "MANIFEST.csv"
        entries = read_manifest(manifest) if manifest.is_file() else {}
        for item in wanted:
            if item.directory == directory:
                entries[item.name] = (item.size, item.sha256)
        write_manifest(manifest, entries)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a config's Lc0 training files")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    if not sys.flags.isolated:
        parser.error("run with python -I")
    try:
        download(load_config(args.config))
    except DownloadError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
