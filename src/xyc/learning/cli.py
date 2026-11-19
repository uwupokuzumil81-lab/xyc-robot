"""The shared learning entry point; selecting ACT only changes the adapter."""

from xyc.robot.profile import robot_profile

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

from .config import load_config
from xyc.paths import project_root


def parser():
    result = argparse.ArgumentParser(description='XYC datasets, model training, inference and explicit motion replay')
    sub = result.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect', help='Inspect committed LeRobot v3 demonstrations')
    inspect.add_argument('--dataset', required=True)
    replay = sub.add_parser('replay', help='Check recorded actions, or execute them on the robot with --execute')
    replay.add_argument('--dataset', required=True)
    replay.add_argument('--episode', type=int, default=0)
    replay.add_argument('--frame', type=int, default=0)
    replay.add_argument('--frames', type=int)
    replay.add_argument(
        '--speed', type=float, default=1.,
        help='0.1..1.0 replay pacing; mobile episodes require 1.0 because speeds are fixed',
    )
    replay.add_argument('--host', default='localhost')
    replay.add_argument('--port', type=int, default=robot_profile().ports['motion'])
    replay.add_argument('--execute', action='store_true', help='Connect and physically replay all 18 dimensions')
    replay.add_argument('--home-lift', action='store_true', help='Permit downward lift homing if the reference is missing')
    replay.add_argument('--output', help='New directory for replay execution logs')
    for name in ('train', 'smoke'):
        train = sub.add_parser(name, help='Train a model' if name == 'train' else 'Short CPU test using adapter-provided dimensions')
        train.add_argument('--config', default=str(project_root() / 'config/train/act.yaml'))
        train.add_argument('--dataset')
        train.add_argument('--output')
        train.add_argument('--device')
        train.add_argument('--steps', type=int)
        train.add_argument('--batch-size', type=int)
        train.add_argument('--workers', type=int)
        train.add_argument('--no-pretrained', action='store_true')
        if name == 'train':
            train.add_argument('--resume', help='Trusted project checkpoint directory or its run directory')
    infer = sub.add_parser('infer', help='Predict a canonical 18-D action chunk on a recorded observation')
    infer.add_argument('--checkpoint', required=True)
    infer.add_argument('--checkpoint-selection', choices=('latest', 'best'), default='latest')
    infer.add_argument('--dataset', required=True)
    infer.add_argument('--episode', type=int, default=0)
    infer.add_argument('--frame', type=int, default=0)
    infer.add_argument('--frames', type=int, default=1, help='Replay this many recorded observations through the action queue')
    infer.add_argument('--device', default='cpu')
    infer.add_argument('--output', required=True)
    infer.add_argument('--record-dir', help='New exclusive run directory; default outputs/inference/<unique-run>')
    infer.add_argument('--record-video', action='store_true', help='Save RGB model-input videos; otherwise reference dataset originals')
    infer.add_argument('--no-record', action='store_true', help='Disable inference trace; keep --output JSON')
    deploy = sub.add_parser('deploy', help='Run a checkpoint on live cameras and the unified real robot controller')
    deploy.add_argument('--checkpoint', required=True)
    deploy.add_argument('--checkpoint-selection', choices=('latest', 'best'), default='latest')
    deploy.add_argument('--host', default='localhost')
    deploy.add_argument('--port', type=int, default=robot_profile().ports['motion'])
    deploy.add_argument('--camera-url', default=f"http://localhost:{robot_profile().ports['data']}")
    deploy.add_argument('--device', default='cuda')
    deploy.add_argument('--home-lift', action='store_true')
    deploy.add_argument('--steps', type=int, help='Stop after this many accepted commands; default runs until Ctrl+C')
    deploy.add_argument('--strategy', choices=('sync_chunk', 'rtc'), default='sync_chunk',
                        help='Use synchronous chunks or background RTC with a compatible adapter')
    deploy.add_argument('--action-steps', type=int,
                        help='Execute this many actions per predicted sync chunk; default uses the checkpoint setting')
    deploy.add_argument('--task', help='Task instruction for policies requesting language input')
    deploy.add_argument('--record-dir', help='New exclusive live inference trace directory')
    deploy.add_argument('--record-video', action='store_true', help='Save aligned live RGB observation videos')
    deploy.add_argument('--no-record', action='store_true', help='Disable the default live inference trace')
    deploy.add_argument('--auto-start', action='store_true', help='Skip manual preparation and immediately start model control')
    deploy.add_argument('--prepare-mode', choices=('keyboard', 'teleop'), default='keyboard',
                        help='keyboard: base/lift only; teleop: leader arms plus keyboard base/lift')
    return result


def main(argv=None):
    os.environ.setdefault('TORCH_HOME', str(project_root() / '.setup/torch'))
    args = parser().parse_args(argv)
    try:
        if args.command == 'replay':
            from .replay import ReplayPlan, execute
            plan = ReplayPlan(args.dataset, episode=args.episode, frame=args.frame, frames=args.frames, speed=args.speed)
            value = (execute(plan, host=args.host, port=args.port, home_lift=args.home_lift, output_dir=args.output)
                     if args.execute else plan.summary())
        elif args.command == 'inspect':
            from .dataset_metadata import DatasetMetadata
            metadata = DatasetMetadata(args.dataset)
            value = {'root': str(metadata.root), 'schema': metadata.schema.to_dict(),
                     'episodes': {i: ep.length for i, ep in metadata.episodes.items()},
                     'frames': sum(ep.length for ep in metadata.episodes.values()),
                     'action_semantics': metadata.assets.get('action_semantics')}
        elif args.command in ('train', 'smoke'):
            from .trainer import train
            config = load_config(args.config)
            if args.command == 'smoke':
                config = replace(config, device='cpu', precision='float32', steps=2, batch_size=1, warmup_steps=0,
                                 num_workers=0, save_every=2, eval_every=2, eval_batches=1, log_every=1,
                                 output_dir=str(project_root() / 'outputs' / (config.policy + '-smoke')))
            updates = {}
            for argument, field in (('dataset', 'dataset'), ('output', 'output_dir'), ('device', 'device'),
                                    ('steps', 'steps'), ('batch_size', 'batch_size'), ('workers', 'num_workers')):
                item = getattr(args, argument)
                if item is not None:
                    updates[field] = str(Path(item).expanduser().resolve()) if field in ('dataset', 'output_dir') else item
            config = replace(config, **updates)
            if args.no_pretrained or args.command == 'smoke':
                from .registry import policy_adapter
                config.model = policy_adapter(config.policy).training_options(
                    config.model, smoke=args.command == 'smoke', no_pretrained=args.no_pretrained)
            value = train(config, resume=getattr(args, 'resume', None))
            if args.command == 'smoke':
                from .inference import predict_dataset
                prediction = predict_dataset(value['checkpoint'], config.dataset, record=False,
                                             output=Path(config.output_dir) / 'prediction.json')
                value['prediction_shape'] = prediction['shape']
                value['note'] = 'Reduced policy pipeline test; not a task-trained deployable policy'
        elif args.command == 'deploy':
            from .deploy import deploy
            value = deploy(args.checkpoint, host=args.host, port=args.port,
                           camera_url=args.camera_url, device=args.device,
                           home_lift=args.home_lift, steps=args.steps, strategy=args.strategy,
                           action_steps=args.action_steps,
                           task=args.task, record=not args.no_record, record_dir=args.record_dir,
                           record_video=args.record_video, interactive=not args.auto_start,
                           prepare_mode=args.prepare_mode, checkpoint_selection=args.checkpoint_selection)
        else:
            from .inference import predict_dataset
            value = predict_dataset(args.checkpoint, args.dataset, episode=args.episode, frame=args.frame,
                                    frames=args.frames, device=args.device, output=args.output,
                                    record=not args.no_record, record_dir=args.record_dir,
                                    record_video=args.record_video, checkpoint_selection=args.checkpoint_selection)
            value = {key: item for key, item in value.items() if key != 'actions'}
        print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print('学习模块：' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
