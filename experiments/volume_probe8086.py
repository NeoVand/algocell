"""CPU-only, 60-second real-volume preparation probe; no science inspection."""
import json
import os
import time
import uuid
from pathlib import Path
import modal

app = modal.App('s2-volume-operations-probe')
volume = modal.Volume.from_name('algocell-atlas-runs')

@app.function(image=modal.Image.from_id('im-7cqReP44xe1Ns0lCbVrLOS'),
              cpu=(.125, .125), memory=(256, 256), timeout=60,
              startup_timeout=120, retries=0, single_use_containers=True,
              max_containers=1, min_containers=0, buffer_containers=0,
              volumes={'/vol': volume})
def probe(archive=False, publication_source=''):
    import fcntl
    import gzip
    import hashlib
    started = time.monotonic()
    root = Path('/vol/s8086_volume_probes') / uuid.uuid4().hex
    results = {}
    def operation(name, fn):
        try:
            value = fn()
            results[name] = {'supported': True, 'result': value}
        except Exception as exc:
            results[name] = {'supported': False, 'error': type(exc).__name__,
                             'errno': getattr(exc, 'errno', None), 'reason': str(exc)}
    operation('volume.reload.initial', volume.reload)
    operation('mkdir.parents', lambda: root.mkdir(parents=True, exist_ok=False))
    operation('mkdir.exist_ok', lambda: root.mkdir(parents=True, exist_ok=True))
    source = root / 'source'
    def create():
        with source.open('xb') as f:
            f.write(b'fixed probe bytes'); f.flush()
        return source.stat().st_size
    operation('exclusive_create.write.flush.close', create)
    def exclusive():
        try:
            with source.open('xb'): pass
        except FileExistsError: return 'existing file rejected'
        raise AssertionError('exclusive create clobbered existing file')
    operation('exclusive_create.no_clobber', exclusive)
    operation('read_bytes', lambda: source.read_bytes().hex())
    operation('exists.stat.iterdir', lambda: [source.exists(), source.stat().st_size, len(list(root.iterdir()))])
    def append():
        with source.open('ab') as f: f.write(b' appended'); f.flush()
    operation('append.flush', append)
    def fsync():
        with source.open('rb') as f: os.fsync(f.fileno())
    operation('file.fsync', fsync)
    def flock():
        with source.open('rb') as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_UN)
    operation('flock.local.only', flock)
    operation('chmod', lambda: os.chmod(source, 0o600))
    operation('link', lambda: os.link(source, root / 'hardlink'))
    operation('rename', lambda: os.rename(source, root / 'renamed'))
    operation('replace', lambda: os.replace(root / 'renamed', root / 'replaced'))
    def directory_fsync():
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    operation('directory.open.fsync.close', directory_fsync)
    def compressed():
        with gzip.open(root / 'compressed.gz', 'xt') as f: f.write('fixed gzip probe')
        with gzip.open(root / 'compressed.gz', 'rt') as f: return f.read()
    operation('gzip.exclusive_write.read', compressed)
    operation('volume.commit', volume.commit)
    operation('volume.reload.after_commit', volume.reload)
    operation('durable_read_after_reload', lambda: (root / 'replaced').read_bytes().hex())
    operation('unlink', lambda: (root / 'replaced').unlink())
    def rmdir():
        directory = root / 'empty'; directory.mkdir(); directory.rmdir()
    operation('mkdir.rmdir', rmdir)
    if publication_source:
        namespace = {'__name__': 'verified_recovery8086'}
        exec(compile(publication_source, 'recovery8086.py', 'exec'), namespace)
        from s8086 import Registration
        registration = Registration(250000, ('A',))
        def fixture(name, attempt_id):
            attempt, _ = namespace['attempt_directory'](root, name, attempt_id=attempt_id)
            rows = [{'kind': 'sample', 'step': s} for s in registration.sample_steps]
            for step in registration.snapshots:
                filename = name + '.t' + str(step)
                (attempt / filename).write_bytes(b'artificial probe snapshot')
                rows.append({'kind': 'snapshot', 'step': step, 'file': filename})
            rows.append({'kind': 'summary', 'steps_run': registration.horizon})
            (attempt / (name + '.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            return attempt
        def publication():
            attempt = fixture('artificial', 'first')
            result = namespace['publish_complete'](attempt, root, 'artificial', registration)
            volume.commit(); volume.reload()
            assert (root / 'artificial.jsonl').read_bytes() == (attempt / 'artificial.jsonl').read_bytes()
            return result
        operation('publication.actual_repaired_code.commit_reload', publication)
        def duplicate():
            before = (root / 'artificial.jsonl').read_bytes()
            attempt = fixture('artificial', 'duplicate')
            namespace['publish_complete'](attempt, root, 'artificial', registration)
            assert (root / 'artificial.jsonl').read_bytes() == before
            return 'canonical bytes unchanged'
        operation('publication.duplicate_no_clobber', duplicate)
        def stale():
            attempt = fixture('stale', 'interrupted')
            (root / 'stale.jsonl.publish-reserved').touch(exist_ok=False)
            try: namespace['publish_complete'](attempt, root, 'stale', registration)
            except FileExistsError:
                assert not (root / 'stale.jsonl').exists()
                return 'stale reservation fails closed'
            raise AssertionError('stale reservation ignored')
        operation('publication.stale_marker_no_clobber', stale)
    archived = None
    if archive:
        required = ('rename', 'replace', 'file.fsync', 'exclusive_create.no_clobber',
                    'volume.commit', 'volume.reload.after_commit')
        if not all(results[k]['supported'] for k in required):
            raise RuntimeError('required archival operations unsupported')
        base = Path('/vol/s8086/A'); world = 's8086_A_s8086001'
        # Hash opaque bytes only. Never parse records or decompress snapshots.
        def integrity(path):
            digest = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''): digest.update(chunk)
            return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
        publishing = integrity(base / (world + '.jsonl.publishing'))
        attempts = [p for p in (base / 'attempts' / world).iterdir()
                    if p.is_dir() and (p / (world + '.jsonl')).exists()
                    and integrity(p / (world + '.jsonl')) == publishing]
        if len(attempts) != 1: raise ValueError('archive requires one digest-matched failed attempt')
        attempt = attempts[0]
        if (base / (world + '.jsonl')).exists(): raise ValueError('unexpected published trajectory; stop')
        destination = attempt / 'publication_failure_archive'
        destination.mkdir(exist_ok=False)
        prior = {str(p.relative_to(attempt)): integrity(p) for p in attempt.rglob('*') if p.is_file()}
        moved = []
        for path in sorted(base.glob(world + '.*')):
            before = integrity(path); target = destination / path.name
            os.rename(path, target)
            if integrity(target) != before: raise AssertionError('archive bytes changed')
            moved.append({'source': str(path), 'destination': str(target), **before})
        if not any(x['source'].endswith('.jsonl.publishing') for x in moved):
            raise AssertionError('failed publication record missing')
        for path, metadata in prior.items():
            if integrity(attempt / path) != metadata: raise AssertionError('attempt bytes changed')
        volume.commit(); volume.reload()
        for item in moved:
            if integrity(Path(item['destination'])) != {k:item[k] for k in ('bytes','sha256')}:
                raise AssertionError('archive not durable')
        archived = {'attempt': str(attempt), 'original_attempt_integrity': prior,
                    'moved_unchanged': moved, 'scientific_content_inspected': False,
                    'selection': 'discard infrastructure failure, restart step 0 same seed'}
    result = {'path': str(root), 'operations': results, 'archive': archived,
              'wall_s': time.monotonic() - started, 'cpu': .125, 'memory_mib': 256,
              'server_timeout_s': 60, 'gpu': None,
              'locking_scope': 'flock/exclusive create are local mount operations; no distributed lock claim'}
    if publication_source: result['publication_source_sha256'] = hashlib.sha256(publication_source.encode()).hexdigest()
    with (root / 'record.json').open('x') as f: json.dump(result, f)
    volume.commit()
    return result
