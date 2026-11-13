"""Persistent local logging with workspace links and post-stop verified copies.

Only disk destinations change. No records are dropped, and the controller's
bounded journal queue and fail-closed policy remain in effect.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

LOCAL_LOG_ENV = 'FINSSIM_LOCAL_LOG_ROOT'
DEFAULT_LOCAL_ROOT = Path.home() / '.local/state/finssim/control_logs'


def filesystem_type(path):
    path = Path(path).resolve()
    matches = []
    for line in Path('/proc/mounts').read_text().splitlines():
        fields = line.split()
        mount = Path(fields[1].replace('\\040', ' '))
        if path.is_relative_to(mount): matches.append((len(mount.parts), fields[2]))
    return max(matches)[1] if matches else 'unknown'


def prepare_local_logs(directory, streams, *, directories=(), local_root=None):
    """Called before arming; existing destinations are never overwritten."""
    directory = Path(directory).resolve()
    configured = local_root or os.environ.get(LOCAL_LOG_ENV)
    if not configured: return directory
    names = tuple(streams) + tuple(directories)
    if any(Path(n).name != n or n in ('.', '..') for n in names):
        raise ValueError('Expected log basenames')
    if any((directory/n).exists() or (directory/n).is_symlink() for n in names):
        raise FileExistsError('Preserve existing logs; choose a new session')
    local_root = Path(configured).expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    fs = filesystem_type(local_root)
    if fs not in ('ext4', 'xfs', 'btrfs', 'zfs'):
        raise ValueError('Control log storage needs a persistent local filesystem: ' + fs)
    if shutil.disk_usage(local_root).free < 2 * 1024**3:
        raise OSError('Less than 2 GiB available for persistent control logs')
    key = hashlib.sha256(str(directory).encode()).hexdigest()[:24]
    target = local_root/key
    target.mkdir(exist_ok=False)
    for name in directories: (target/name).mkdir()
    metadata = dict(directory=str(directory), local_directory=str(target),
                    streams=list(streams), directories=list(directories),
                    filesystem=fs, state='local',
                    policy='lossless local journal; bounded fail-closed queue; verified workspace archive after stop; local originals retained')
    encoded = json.dumps(metadata, indent=2)+'\n'
    (target/'log_storage.json').write_text(encoded)
    # All links and metadata are established before any command can be enabled.
    for name in names:
        (directory/name).symlink_to(target/name, target_is_directory=name in directories)
    (directory/'log_storage.json').write_text(encoded)
    return target


def _digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def _tree_digest(path):
    return {str(p.relative_to(path)): _digest(p) for p in path.rglob('*') if p.is_file()}


def archive_local_logs(directory):
    """After controller exit only. Copy/verify before replacing workspace links.

    SSD originals remain recoverable even if workspace copying fails.
    """
    directory = Path(directory).resolve()
    manifest = directory/'log_storage.json'
    if not manifest.exists(): return None
    metadata = json.loads(manifest.read_text())
    target = Path(metadata['local_directory'])
    archived = dict(metadata.get('archived_sha256', {}))
    for name in metadata['streams'] + metadata['directories']:
        source, dest = target/name, directory/name
        if not source.exists(): continue  # e.g. never enabled, no heading captured
        if not dest.is_symlink():
            expected = _tree_digest(source) if source.is_dir() else _digest(source)
            actual = _tree_digest(dest) if dest.is_dir() else _digest(dest)
            if actual != expected: raise OSError('Workspace archive differs: '+name)
            archived[name] = expected
            continue
        if dest.resolve() != source.resolve(): raise ValueError('Log link destination changed')
        temporary = directory/('.'+name+'.archive-'+uuid.uuid4().hex)
        if source.is_dir():
            shutil.copytree(source, temporary)
            expected = _tree_digest(source)
            if _tree_digest(temporary) != expected: raise OSError('Archive verification failed: '+name)
            backup = directory/('.'+name+'.ssd-link-'+uuid.uuid4().hex)
            dest.rename(backup)
            try: temporary.rename(dest)
            except BaseException:
                backup.rename(dest)
                raise
            backup.unlink()
        else:
            shutil.copy2(source, temporary)
            expected = _digest(source)
            if _digest(temporary) != expected: raise OSError('Archive verification failed: '+name)
            os.replace(temporary, dest)
        archived[name] = expected
    metadata.update(state='archived', archived_sha256=archived, local_originals_retained=True)
    manifest.write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata
