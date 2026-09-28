"""Bounded, regular-file-only workspace transfers (standard library only)."""

import io
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile

MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_WORKSPACE_BYTES = 512 * 1024 * 1024
MAX_MEMBERS = 10_000


def safe_path(root, name):
    """Resolve a relative archive path without traversing existing symlinks."""
    root = Path(root).resolve()
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe workspace path: {name!r}")
    target = root
    for part in path.parts:
        target = target / part
        if target.is_symlink():
            raise ValueError(f"Workspace symlinks are not supported: {name!r}")
    if target == root or not target.resolve().is_relative_to(root):
        raise ValueError(f"Unsafe workspace path: {name!r}")
    return target


def extract_workspace(data, root):
    """Validate the whole archive before writing; never extract links/devices."""
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("Workspace archive exceeds transfer limit")
    root = Path(root).resolve()
    members = []
    total = 0
    names = set()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive:
            if len(members) >= MAX_MEMBERS:
                raise ValueError("Too many workspace entries")
            if not (member.isfile() or member.isdir()) or member.size < 0:
                raise ValueError(f"Unsupported workspace entry: {member.name!r}")
            target = safe_path(root, member.name)
            if target in names:
                raise ValueError(f"Duplicate workspace entry: {member.name!r}")
            names.add(target)
            total += member.size
            if total > MAX_WORKSPACE_BYTES:
                raise ValueError("Expanded workspace exceeds size limit")
            members.append((member, target))
        for member, target in members:
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
    return {m.name for m, _ in members if m.isfile()}


def pack_workspace(root, max_file_mb):
    """Return (archive, transferred names, skipped names), excluding links."""
    root = Path(root).resolve()
    limit = max_file_mb * 1024 * 1024
    buffer = io.BytesIO()
    sent, skipped = set(), []
    total = 0
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()]
            for name in sorted(files):
                path = Path(directory) / name
                if path.is_symlink() or not path.is_file():
                    continue
                rel = path.relative_to(root).as_posix()
                size = path.stat().st_size
                if size > limit:
                    skipped.append(rel)
                    continue
                total += size
                if total > MAX_WORKSPACE_BYTES or len(sent) >= MAX_MEMBERS:
                    raise ValueError("Workspace exceeds transfer limits")
                archive.add(path, arcname=rel, recursive=False)
                sent.add(rel)
    data = buffer.getvalue()
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("Compressed workspace exceeds transfer limit")
    return data, sent, skipped
