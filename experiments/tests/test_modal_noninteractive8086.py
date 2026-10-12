"""Exercise the real frozen stop subprocess chain without a terminal or provider."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import modal
import pytest

from s8086 import HERE, PRE


@pytest.fixture
def fake_modal(tmp_path, monkeypatch):
    package = tmp_path / 'modal'
    package.mkdir()
    (package / '__init__.py').write_text('')
    # Production uses python -m modal. Forward that unmodified invocation to
    # an actual fake modal executable, keeping its inherited detached streams.
    (package / '__main__.py').write_text(
        "import os, shutil, sys\nos.execv(shutil.which('modal'), ['modal', *sys.argv[1:]])\n")
    executable = tmp_path / 'bin' / 'modal'
    executable.parent.mkdir()
    executable.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
args = sys.argv[1:]
tty = [os.isatty(fd) for fd in (0, 1, 2)]
accepted = args[:2] == ['app', 'stop'] and ('--yes' in args or '-y' in args)
with open(os.environ['FAKE_MODAL_LOG'], 'a') as f:
    f.write(json.dumps({'args': args, 'tty': tty, 'accepted': accepted,
                        'pid': os.getpid(), 'session': os.getsid(0)}) + '\\n')
if any(tty):
    sys.exit(91)
if not accepted:
    print('Aborted: no interactive terminal detected. Rerun with --yes (-y) to skip confirmation.', file=sys.stderr)
    sys.exit(1)
''')
    executable.chmod(0o755)
    log = tmp_path / 'calls.jsonl'
    monkeypatch.setenv('PATH', str(executable.parent) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('PYTHONPATH', str(tmp_path) + os.pathsep + str(HERE))
    monkeypatch.setenv('FAKE_MODAL_LOG', str(log))
    return log


def detached(command):
    return subprocess.run(command, stdin=subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          start_new_session=True, timeout=20)


def test_fake_rejects_missing_confirmation_without_tty(fake_modal):
    result = detached([sys.executable, '-m', 'modal', 'app', 'stop', 'ap-fake'])
    assert result.returncode == 1
    row = json.loads(fake_modal.read_text())
    assert row['tty'] == [False, False, False] and not row['accepted']


@pytest.mark.parametrize('path', ['entrypoint', 'provider_stop'])
def test_each_frozen_stop_path_detached(fake_modal, path):
    if path == 'entrypoint':
        command = [sys.executable, str(HERE / 'lifecycle8086.py'), '--stop', 'ap-fake']
    else:
        command = [sys.executable, '-c', '''from lifecycle8086 import ModalProvider
p = ModalProvider.__new__(ModalProvider)
p.controls = []
p.stop('ap-fake')
assert len(p.controls) == 1
assert p.controls[0].wait(timeout=15) == 0
''']
    assert detached(command).returncode == 0
    rows = [json.loads(line) for line in fake_modal.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]['args'] == ['app', 'stop', '--yes', 'ap-fake']
    assert rows[0]['accepted'] and rows[0]['tty'] == [False, False, False]


def test_all_frozen_modal_cli_paths_are_covered():
    paths = []
    for name in json.loads((PRE / 'freeze.json').read_text())['files']:
        if not name.endswith('.py') or name.startswith('tests/'):
            continue
        tree = ast.parse((HERE / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.List, ast.Tuple)):
                values = [item.value for item in node.elts if isinstance(item, ast.Constant)]
                if 'modal' in values:
                    paths.append((name, values))
    assert paths == [('lifecycle8086.py', ['-m', 'modal', 'app', 'stop', '--yes'])]


def test_real_pinned_cli_accepts_confirmation_flag_without_provider():
    assert modal.__version__ == '1.6.1'
    result = subprocess.run([sys.executable, '-m', 'modal', 'app', 'stop', '--yes', '--help'],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            start_new_session=True, timeout=20)
    assert result.returncode == 0
    assert re.search(r'-y,\s+--yes\s+Run without pausing for confirmation', result.stdout)
