"""One lazy policy lookup, without importing GPU libraries on the robot."""


def make_policy(name, *, model_config, schema, statistics, initialize_pretrained=True):
    if name != 'act':
        raise ValueError(f'Policy {name!r} is not registered; available policies: act')
    from xyc.models.act.adapter import ACTAdapter
    return ACTAdapter(model_config, schema, statistics, initialize_pretrained=initialize_pretrained)
