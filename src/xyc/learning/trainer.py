"""Common optimizer, validation, checkpoint and resume loop for policy adapters."""

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from .checkpoint import load_checkpoint, refresh_checkpoint_indexes, save_checkpoint
from xyc.io import write_json
from .dataset import EpisodeDataset
from .dataset_metadata import DatasetMetadata
from .registry import make_policy, policy_adapter
from xyc.locking import acquire_file_lock


class StepBatches(Sampler):
    """Sampling depends on the step, so worker prefetch cannot move a resume cursor."""
    def __init__(self, length, batch_size, start, end, seed):
        self.length, self.batch_size = length, batch_size
        self.start, self.end, self.seed = start, end, seed

    def __iter__(self):
        for step in range(self.start, self.end):
            generator = torch.Generator().manual_seed(self.seed + step)
            yield torch.randint(self.length, (self.batch_size,), generator=generator).tolist()

    def __len__(self):
        return self.end - self.start


def worker_init(_):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'].cpu())
    if value['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in value['cuda']])


def configure_device(config):
    device = torch.device(config.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this Python environment; use --device cpu for a smoke test')
    if config.precision == 'bfloat16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('bfloat16 requires a CUDA GPU supporting BF16; use float32 otherwise')
    torch.set_num_threads(config.cpu_threads)
    cv2.setNumThreads(1)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    return device


def validate_resume(config, previous):
    # Paths may change when moving a run to a cloud machine; content is checked separately.
    allowed = {'dataset', 'output_dir', 'steps', 'save_every', 'eval_every', 'eval_batches',
               'log_every', 'num_workers', 'device', 'cpu_threads'}
    current = config.to_dict()
    for key in set(previous) | set(current):
        if key not in allowed and current.get(key) != previous.get(key):
            raise ValueError('Resume cannot change training setting: ' + key)
    if config.scheduler == 'cosine' and config.steps != previous['steps']:
        raise ValueError('Cosine resume must retain the original total step count')


def train(config, *, resume=None, progress=print):
    config.validate()
    device = configure_device(config)
    metadata = DatasetMetadata(config.dataset, config.cameras)
    train_ids, val_ids = metadata.split(validation_fraction=config.validation_fraction, seed=config.seed,
                                       train_episodes=config.train_episodes,
                                       validation_episodes=config.validation_episodes)
    fingerprint = metadata.fingerprint(train_ids + val_ids)
    root = Path(config.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.training.lock'
    try:
        descriptor = acquire_file_lock(lock)
    except OSError as exc:
        raise FileExistsError('Another trainer owns this run: ' + str(root)) from exc
    dataset = validation = None
    try:
        if not resume and ((root / 'run.json').exists() or list(root.glob('checkpoint-*'))):
            raise FileExistsError('Run already exists: choose a new output_dir or explicitly resume it')
        validation_spec = policy_adapter(config.policy).validation_spec()
        start, best_metric, saved_state = 0, validation_spec.initial_best(), None
        if resume:
            adapter, manifest, saved_path = load_checkpoint(resume, device=device)
            validate_resume(config, manifest['train_config'])
            if (manifest['dataset_fingerprint'] != fingerprint
                    or json.dumps(manifest['schema'], sort_keys=True) != json.dumps(metadata.schema.to_dict(), sort_keys=True)
                    or manifest['train_episodes'] != train_ids or manifest['validation_episodes'] != val_ids):
                raise ValueError('Resume dataset, schema or episode split differs from the checkpoint')
            if (manifest.get('validation_spec') is not None
                    and manifest['validation_spec'] != validation_spec.to_dict()):
                raise ValueError('Resume cannot change the validation definition')
            refresh_checkpoint_indexes(root, validation_spec)
            # Only load optimizer/RNG pickle from this project's trusted training runs.
            saved_state = torch.load(saved_path / 'training.pt', map_location='cpu', weights_only=False)
            start, best_metric = manifest['step'], saved_state['best_metric']
        else:
            factory = policy_adapter(config.policy)
            statistics = metadata.statistics(train_ids, factory.statistics_spec(config.model, metadata.schema))
            adapter = make_policy(config.policy, model_config=config.model, schema=metadata.schema,
                                  statistics=statistics).to(device)
        if config.steps <= start:
            raise ValueError('Total steps must be greater than the checkpoint step')
        model_options = adapter.model_config()
        optimizer = torch.optim.AdamW(adapter.optimizer_groups(learning_rate=config.learning_rate,
                                     backbone_learning_rate=config.backbone_learning_rate),
                                     weight_decay=config.weight_decay)

        def lr_scale(step):
            if step < config.warmup_steps:
                return (step + 1) / config.warmup_steps
            if config.scheduler == 'constant':
                return 1.
            fraction = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
            return .5 * (1 + math.cos(math.pi * min(1., fraction)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        if saved_state:
            optimizer.load_state_dict(saved_state['optimizer'])
            # AdamW state belongs on each parameter's device after a CPU-loaded resume.
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value) and key != 'step':
                        state[key] = value.to(device)
            scheduler.load_state_dict(saved_state['scheduler'])
            restore_rng(saved_state['rng'])
        sample_spec = adapter.sample_spec(config.model, metadata.schema)
        dataset = EpisodeDataset(metadata, train_ids, spec=sample_spec)
        validation = EpisodeDataset(metadata, val_ids, spec=sample_spec)
        loader = DataLoader(dataset, batch_sampler=StepBatches(len(dataset), config.batch_size, start,
                                                             config.steps, config.seed),
                            num_workers=config.num_workers, worker_init_fn=worker_init,
                            generator=torch.Generator().manual_seed(config.seed),
                            pin_memory=device.type == 'cuda')
        # Spread a bounded validation budget across entire episodes, not only their opening frames.
        validation_indices = np.linspace(0, max(0, len(validation) - 1),
                                         min(len(validation), config.eval_batches * config.batch_size), dtype=int)
        val_loader = DataLoader(Subset(validation, validation_indices.tolist()),
                                batch_size=config.batch_size, num_workers=0,
                                generator=torch.Generator().manual_seed(config.seed + 1)) if val_ids else None
        base_manifest = {'policy': config.policy, 'model_config': model_options,
                         'sample_spec': sample_spec.to_dict(), 'validation_spec': validation_spec.to_dict(),
                         'statistics_spec': adapter.statistics_spec(config.model, metadata.schema).to_dict(),
                         'schema': metadata.schema.to_dict(), 'statistics': adapter.statistics,
                         'robot_assets': metadata.assets, 'dataset_fingerprint': fingerprint,
                         'train_episodes': train_ids, 'validation_episodes': val_ids,
                         'train_config': config.to_dict()}
        write_json(root / 'run.json', base_manifest)
        progress(f'Train {train_ids}, validation {val_ids}; {len(dataset)} frames; '
                 f'{sum(p.numel() for p in adapter.parameters()):,} parameters on {device}; steps {start}->{config.steps}')
        started, last_time = time.perf_counter(), time.perf_counter()
        last_validation, checkpoint_path, validation_values = None, None, {}
        with (root / 'metrics.jsonl').open('a', encoding='utf-8') as log:
            for step, batch in enumerate(loader, start=start + 1):
                fetched = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                context = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if config.precision == 'bfloat16' else nullcontext()
                with context:
                    loss, terms = adapter.compute_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss at step {step}')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), config.grad_clip_norm,
                                                      error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                computed = time.perf_counter()
                record = {'step': step, 'loss': float(loss.detach()), **terms, 'grad_norm': float(norm),
                          'learning_rate': optimizer.param_groups[0]['lr'],
                          'data_seconds': fetched - last_time, 'compute_seconds': computed - fetched}
                improved = False
                step_validation = {}
                if val_loader and (step % config.eval_every == 0 or step == config.steps):
                    totals, weights = {}, {}
                    for index, val_batch in enumerate(val_loader):
                        metrics = adapter.validation_metrics(val_batch)
                        if validation_spec.primary not in metrics:
                            raise ValueError('Adapter omitted its declared primary validation metric')
                        for name, metric in metrics.items():
                            totals[name] = totals.get(name, 0.) + metric.value * metric.weight
                            weights[name] = weights.get(name, 0.) + metric.weight
                        if index + 1 >= config.eval_batches:
                            break
                    validation_values = {name: total / weights[name] for name, total in totals.items()}
                    last_validation = validation_values[validation_spec.primary]
                    step_validation = {'validation_' + name: value for name, value in validation_values.items()}
                    record.update(step_validation)
                    improved = validation_spec.improved(last_validation, best_metric)
                    if improved:
                        best_metric = last_validation
                if step % config.log_every == 0 or step == start + 1 or step == config.steps:
                    log.write(json.dumps(record, allow_nan=False) + '\n')
                    log.flush()
                    progress(f"step {step}/{config.steps} loss={record['loss']:.5f} "
                             f"data={record['data_seconds']:.3f}s compute={record['compute_seconds']:.3f}s")
                if step % config.save_every == 0 or step == config.steps or improved:
                    state = {'step': step, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                             'rng': rng_state(), 'best_metric': best_metric}
                    checkpoint_path = save_checkpoint(root, adapter, {**base_manifest, 'step': step, **step_validation}, state)
                last_time = time.perf_counter()
        result = {'step': config.steps, 'checkpoint': str(checkpoint_path),
                  'elapsed_seconds': time.perf_counter() - started, 'last_loss': record['loss'],
                  **{'validation_' + name: value for name, value in validation_values.items()}}
        write_json(root / 'result.json', result)
        return result
    finally:
        if dataset is not None:
            dataset.close()
        if validation is not None:
            validation.close()
        os.close(descriptor)
