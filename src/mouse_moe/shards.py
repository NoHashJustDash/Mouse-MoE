# SPDX-License-Identifier: GPL-3.0-or-later
"""Streaming access to Lc0 training tars (one gzipped game per member)."""

from __future__ import annotations

import gzip
import hashlib
import tarfile
import zlib
from collections.abc import Container, Iterator
from pathlib import Path


class ShardError(ValueError):
    """A training shard is unreadable or contains a corrupt member."""


def count_games(path: Path) -> int:
    """Number of game members, read from tar headers only (nothing is decompressed)."""
    try:
        with tarfile.open(path, mode="r:") as archive:
            return sum(1 for m in archive.getmembers() if m.isfile() and m.name.endswith(".gz"))
    except tarfile.TarError as error:
        raise ShardError(f"{path}: unreadable tar archive") from error


def stream_members(archive: tarfile.TarFile) -> Iterator[tarfile.TarInfo]:
    """Iterate a streaming (``r|``) archive without keeping the headers already read.

    TarFile records every TarInfo it passes in ``members`` (measured: 13.2 MiB per
    20,000 headers, ~50 MB per Stage A shard). The loader keeps 22 shards open in
    each of 6 workers, so the cache alone grew to several GB and hit the job's
    memory limit at step ~31k of ab_m1_fixed. Stream mode never reads it back.
    """
    for member in archive:
        archive.members.clear()  # type: ignore[attr-defined]  # undeclared in typeshed
        yield member


def iter_games(path: Path, only: Container[int] | None = None) -> Iterator[tuple[str, bytes]]:
    """Yield (member name, decompressed game bytes) in archive order.

    Non-game members (directory entries, anything not ending in ``.gz``) are
    skipped; Stage A tars hold exactly two such entries each. With ``only``, just
    the games at those 0-based game indices are decompressed and yielded.
    """
    try:
        with tarfile.open(path, mode="r|") as archive:
            index = -1
            for member in stream_members(archive):
                if not member.isfile() or not member.name.endswith(".gz"):
                    continue
                index += 1
                if only is not None and index not in only:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    raise ShardError(f"{path}: cannot read member {member.name}")
                compressed = handle.read()
                try:
                    yield member.name, gzip.decompress(compressed)
                except (OSError, EOFError, zlib.error) as error:
                    raise ShardError(f"{path}: corrupt gzip member {member.name}") from error
    except tarfile.TarError as error:
        raise ShardError(f"{path}: unreadable tar archive") from error


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(directory: Path) -> dict[str, tuple[int, str]]:
    """Read ``MANIFEST.csv`` (file,bytes,sha256) written when the data is downloaded."""
    manifest = directory / "MANIFEST.csv"
    if not manifest.is_file():
        raise ShardError(f"missing {manifest}; the data directory is not a verified download")
    entries: dict[str, tuple[int, str]] = {}
    lines = manifest.read_text(encoding="ascii").splitlines()
    for line in lines[1:]:
        name, size, digest = (field.strip('"') for field in line.split(","))
        entries[name] = (int(size), digest)
    return entries


def verified_shards(directory: Path, names: list[str], check_hash: bool = False) -> list[Path]:
    """Resolve shard names against the manifest; sizes always, hashes on request."""
    manifest = load_manifest(directory)
    paths: list[Path] = []
    for name in names:
        if name not in manifest:
            raise ShardError(f"{name} is not in {directory / 'MANIFEST.csv'}")
        path = directory / name
        size, digest = manifest[name]
        if not path.is_file() or path.stat().st_size != size:
            raise ShardError(f"{path} is missing or has the wrong size (expected {size})")
        if check_hash and sha256_file(path) != digest:
            raise ShardError(f"{path} SHA-256 does not match the manifest")
        paths.append(path)
    return paths
