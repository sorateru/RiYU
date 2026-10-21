"""Streaming readers for user-provided Tenhou XML archive bundles.

The Google Drive export used by the project stores one or more ``tar.xz``
members inside a ZIP.  Reading both layers as streams avoids expanding tens of
gigabytes of raw logs on disk.  Nothing in this module downloads data or
extracts archive paths onto the filesystem.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import tarfile
import zipfile
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import BinaryIO


_TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
_COMPRESSED_XML_SUFFIXES = (".xml.gz", ".mjlog.gz", ".xml.bz2", ".mjlog.bz2")


def validate_tenhou_archive(path: str | Path) -> None:
    """Reject incomplete or unsupported bundles before creating output files."""

    archive = Path(path)
    if not archive.is_file():
        raise ValueError(f"archive does not exist: {archive}")
    lower = archive.name.casefold()
    if lower.endswith(".zip"):
        try:
            with zipfile.ZipFile(archive) as bundle:
                if not bundle.infolist():
                    raise ValueError(f"archive is empty: {archive}")
        except zipfile.BadZipFile as error:
            raise ValueError(f"incomplete or invalid ZIP archive: {archive}") from error
        return
    if lower.endswith(_TAR_SUFFIXES):
        try:
            with tarfile.open(archive, mode="r:*"):
                pass
        except tarfile.TarError as error:
            raise ValueError(f"incomplete or invalid tar archive: {archive}") from error
        return
    raise ValueError(f"unsupported Tenhou archive type: {archive}")


def iter_tenhou_archive_xml(
    path: str | Path,
    *,
    max_document_bytes: int = 16 * 1024 * 1024,
) -> Iterator[tuple[str, bytes]]:
    """Yield ``(stable source key, XML bytes)`` without extracting to disk."""

    if max_document_bytes <= 0:
        raise ValueError("max_document_bytes must be positive")
    archive = Path(path)
    lower = archive.name.casefold()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                if member.is_dir():
                    continue
                member_name = member.filename
                member_lower = member_name.casefold()
                if member.flag_bits & 0x1:
                    raise ValueError(f"encrypted archive member is unsupported: {member_name}")
                with bundle.open(member) as source:
                    if member_lower.endswith(_TAR_SUFFIXES):
                        yield from _iter_tar_xml(
                            source,
                            f"{archive.name}:{member_name}",
                            max_document_bytes,
                        )
                    else:
                        payload = _read_candidate(source, member_name, max_document_bytes)
                        if payload is not None:
                            yield f"{archive.name}:{member_name}", payload
        return
    if lower.endswith(_TAR_SUFFIXES):
        with archive.open("rb") as source:
            yield from _iter_tar_xml(source, archive.name, max_document_bytes)
        return
    raise ValueError(f"unsupported Tenhou archive type: {archive}")


def _iter_tar_xml(
    source: BinaryIO,
    parent_key: str,
    max_document_bytes: int,
) -> Iterator[tuple[str, bytes]]:
    try:
        bundle = tarfile.open(fileobj=source, mode="r|*")
        with closing(bundle):
            for member in bundle:
                if not member.isfile():
                    continue
                extracted = bundle.extractfile(member)
                if extracted is None:
                    continue
                with closing(extracted):
                    payload = _read_candidate(
                        extracted,
                        member.name,
                        max_document_bytes,
                        declared_size=member.size,
                    )
                if payload is not None:
                    yield f"{parent_key}:{member.name}", payload
    except (tarfile.TarError, EOFError, lzma.LZMAError) as error:
        raise ValueError(f"invalid nested tar stream {parent_key}: {error}") from error


def _read_candidate(
    source: BinaryIO,
    name: str,
    limit: int,
    *,
    declared_size: int | None = None,
) -> bytes | None:
    if declared_size is not None and declared_size > limit:
        raise ValueError(f"XML document exceeds {limit} bytes: {name}")
    payload = source.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"XML document exceeds {limit} bytes: {name}")
    lower = name.casefold()
    try:
        if lower.endswith((".gz", ".gzip")):
            payload = gzip.decompress(payload)
        elif lower.endswith((".bz2", ".bzip2")):
            payload = bz2.decompress(payload)
        elif lower.endswith((".xz", ".lzma")) and not lower.endswith(_TAR_SUFFIXES):
            payload = lzma.decompress(payload)
    except (OSError, EOFError, lzma.LZMAError) as error:
        if lower.endswith(_COMPRESSED_XML_SUFFIXES):
            raise ValueError(f"cannot decompress XML member {name}: {error}") from error
        return None
    if len(payload) > limit:
        raise ValueError(f"decompressed XML document exceeds {limit} bytes: {name}")
    if not payload.lstrip().startswith(b"<"):
        return None
    return payload
