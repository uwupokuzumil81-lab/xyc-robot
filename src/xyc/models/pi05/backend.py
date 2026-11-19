"""Strict initialization from π0.5 base weights; XYC checkpoints use the shared loader."""

from pathlib import Path

def build_policy(config, *, initialize_pretrained):
    from .modeling_pi05 import PI05Policy

    weights = None
    if initialize_pretrained and config.pretrained is not None:
        weights = _pretrained_weights(config.pretrained, config.pretrained_revision)
    policy = PI05Policy(config)
    if weights is not None:
        _load_base(policy, weights)
    return policy


def _pretrained_weights(source, revision):
    from huggingface_hub import hf_hub_download

    path = Path(source).expanduser()
    if path.is_dir():
        weights = path / 'model.safetensors'
        if not weights.is_file():
            raise ValueError(f'π0.5 pretrained weights not found at {weights}')
        return weights
    if path.is_absolute():
        raise ValueError(f'π0.5 pretrained directory not found: {path}')
    return Path(hf_hub_download(source, 'model.safetensors', revision=revision))


def _load_base(policy, weights):
    from safetensors import safe_open

    # The public base omits the policy's "model." prefix and stores one copy
    # of the tied PaliGemma embedding. Honor the file's alias table.
    def key(name):
        return name if name.startswith('model.') else 'model.' + name

    with safe_open(str(weights), framework='pt') as stored:
        state = {key(name): stored.get_tensor(name) for name in stored.keys()}
        for alias, original in (stored.metadata() or {}).items():
            if key(alias) not in state and key(original) in state:
                state[key(alias)] = state[key(original)]
        policy.load_state_dict(state, strict=True)
