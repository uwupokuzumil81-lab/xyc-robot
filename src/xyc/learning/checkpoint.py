"""Platform-owned checkpoint format: weights, manifest and resumable state."""

from xyc.io import write_json

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from uuid import uuid4

import torch
from safetensors.torch import load_file, save_file

from xyc.contracts import PolicySchema
from .registry import make_policy, policy_adapter
from .data_spec import ValidationSpec

FORMAT = 'xyc.policy.checkpoint.v1'




def committed_checkpoints(root):
    """Return complete commits in step order; staging and indexes are not commits."""
    root = Path(root).resolve()
    result = []
    for path in root.glob('checkpoint-*'):
        suffix = path.name.removeprefix('checkpoint-')
        if not path.is_dir() or not suffix.isdigit():
            continue
        if path.resolve().parent != root or path.is_symlink() or any(
                not (path / name).is_file() for name in ('manifest.json', 'model.safetensors', 'training.pt')):
            raise ValueError('Incomplete or unsafe committed checkpoint: ' + str(path))
        manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('step') != int(suffix):
            raise ValueError('Checkpoint step differs from its committed directory')
        result.append((path, manifest))
    return sorted(result, key=lambda row: row[1]['step'])


def best_committed_checkpoint(commits, validation_spec=None):
    """Select using the declared metric; ties retain the earliest best commit."""
    if not commits:
        return None
    if validation_spec is None:
        latest = commits[-1][1]
        if 'validation_spec' in latest:
            validation_spec = ValidationSpec(**latest['validation_spec'])
        elif latest.get('policy'):
            # Earlier manifests recorded the named scores but not their definition.
            validation_spec = policy_adapter(latest['policy']).validation_spec()
    best = None
    if validation_spec is not None:
        score = validation_spec.initial_best()
        for path, manifest in commits:
            declared = manifest.get('validation_spec')
            if declared is not None and declared != validation_spec.to_dict():
                raise ValueError('Committed checkpoints use different validation definitions')
            value = manifest.get('validation_' + validation_spec.primary)
            if value is not None:
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError('Invalid committed validation metric')
                if validation_spec.improved(value, score):
                    best, score = path, value
    return best


def refresh_checkpoint_indexes(root, validation_spec=None):
    """Rebuild disposable indexes from committed metadata, also after a crash."""
    root = Path(root).resolve()
    commits = committed_checkpoints(root)
    if not commits:
        return
    best = best_committed_checkpoint(commits, validation_spec)
    write_json(root / 'last_checkpoint.json', {'path': commits[-1][0].name})
    if best is not None:
        write_json(root / 'best_checkpoint.json', {'path': best.name})
    else:
        (root / 'best_checkpoint.json').unlink(missing_ok=True)


def save_checkpoint(output_dir, adapter, manifest, training_state):
    root = Path(output_dir).resolve()
    target = root / f'checkpoint-{manifest["step"]:08d}'
    if target.exists():
        raise FileExistsError('Refusing to overwrite checkpoint: ' + str(target))
    stage = root / ('.checkpoint-' + uuid4().hex)
    stage.mkdir()
    try:
        state = {key: tensor.detach().cpu().contiguous().clone() for key, tensor in adapter.state_dict().items()}
        save_file(state, str(stage / 'model.safetensors'))
        torch.save(training_state, stage / 'training.pt')
        result = {**manifest, 'format': FORMAT,
                  'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'weights_sha256': hashlib.sha256((stage / 'model.safetensors').read_bytes()).hexdigest()}
        write_json(stage / 'manifest.json', result)
        os.replace(stage, target)
        refresh_checkpoint_indexes(root)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return target


def resolve_checkpoint(path, *, selection='latest'):
    if selection not in ('latest', 'best'):
        raise ValueError('Checkpoint selection must be latest or best')
    path = Path(path).expanduser().resolve()
    if (path / 'manifest.json').is_file():
        if selection != 'latest':
            raise ValueError('Best checkpoint selection requires a run directory')
        return path
    commits = committed_checkpoints(path)
    if not commits:
        raise ValueError('No committed checkpoint at ' + str(path))
    if selection == 'latest':
        return commits[-1][0]
    best = best_committed_checkpoint(commits)
    if best is None:
        raise ValueError('Run has no committed validation metric for best checkpoint selection')
    return best


def load_checkpoint(path, *, device='cpu', selection='latest'):
    path = resolve_checkpoint(path, selection=selection)
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format') != FORMAT:
        raise ValueError('Unsupported checkpoint format')
    weights = path / 'model.safetensors'
    if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest['weights_sha256']:
        raise ValueError('Checkpoint weights hash mismatch')
    schema = PolicySchema.from_dict(manifest['schema'])
    adapter = make_policy(manifest['policy'], model_config=manifest['model_config'], schema=schema,
                          statistics=manifest['statistics'], initialize_pretrained=False)
    adapter.load_state_dict(load_file(str(weights)), strict=True)
    adapter.to(device).eval()
    return adapter, manifest, path
