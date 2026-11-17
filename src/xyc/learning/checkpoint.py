"""Platform-owned checkpoint format: weights, manifest and resumable state."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import torch
from safetensors.torch import load_file, save_file

from .contracts import PolicySchema
from .registry import make_policy

FORMAT = 'xyc.policy.checkpoint.v1'


def json_write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as file:
        json.dump(value, file, ensure_ascii=False, allow_nan=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def save_checkpoint(output_dir, adapter, manifest, training_state, *, best=False):
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
        json_write(stage / 'manifest.json', result)
        os.replace(stage, target)
        json_write(root / 'last_checkpoint.json', {'path': target.name})
        if best:
            json_write(root / 'best_checkpoint.json', {'path': target.name})
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return target


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if (path / 'manifest.json').is_file():
        return path
    pointer = path / 'last_checkpoint.json'
    if not pointer.is_file():
        raise ValueError('No checkpoint manifest or last_checkpoint.json at ' + str(path))
    target = (path / json.loads(pointer.read_text(encoding='utf-8'))['path']).resolve()
    if target.parent != path or not (target / 'manifest.json').is_file():
        raise ValueError('Invalid checkpoint pointer')
    return target


def load_checkpoint(path, *, device='cpu'):
    path = resolve_checkpoint(path)
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
