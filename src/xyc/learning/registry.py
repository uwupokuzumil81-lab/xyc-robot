"""One lazy policy lookup, without importing GPU libraries on the robot."""

from importlib import import_module


POLICIES = {
    'act': ('xyc.models.act.adapter', 'ACTAdapter'),
    'pi05': ('xyc.models.pi05.adapter', 'PI05Adapter'),
}


def policy_adapter(name):
    if name not in POLICIES:
        raise ValueError(f'Policy {name!r} is not registered; available policies: {", ".join(POLICIES)}')
    module, symbol = POLICIES[name]
    return getattr(import_module(module), symbol)


def make_policy(name, *, model_config, schema, statistics, initialize_pretrained=True):
    return policy_adapter(name)(model_config, schema, statistics,
                                initialize_pretrained=initialize_pretrained)
