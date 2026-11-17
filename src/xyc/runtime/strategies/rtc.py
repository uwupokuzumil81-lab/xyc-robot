"""Common RTC strategy boundary; model-specific RTC remains in its adapter.

RTC scheduling can be shared, but guided action-chunk generation is not a
universal post-processing operation.  A compatible adapter must create an RTC
runtime that understands that model's conditioning and action queue.
"""

from .base import ActionSelection, InferenceStrategy


class RTCStrategy(InferenceStrategy):
    """Delegate RTC scheduling/guidance to an adapter-provided runtime.

    A future Pi0.5 adapter should expose ``create_rtc_runtime(**options)``.  The
    returned object must implement ``reset()``, ``select(...)`` and optionally
    ``close()``.  Keeping this capability explicit prevents ACT from silently
    being labelled RTC when it only supports ordinary chunking/ensembling.
    """

    name = 'rtc'

    def __init__(self, adapter, **options):
        factory = getattr(adapter, 'create_rtc_runtime', None)
        if factory is None:
            raise ValueError(
                f'{getattr(adapter, "name", type(adapter).__name__)} does not support RTC; '
                'use sync_chunk or add a model-specific RTC backend')
        self.runtime = factory(**options)
        for method in ('reset', 'select'):
            if not callable(getattr(self.runtime, method, None)):
                raise TypeError(f'Model RTC runtime must implement {method}()')
        self.reset()

    def reset(self):
        self.runtime.reset()

    def select(self, batch, *, observation, predict, now_ns):
        selected = self.runtime.select(
            batch, observation=observation, predict=predict, now_ns=now_ns)
        if not isinstance(selected, ActionSelection):
            raise TypeError('Model RTC runtime must return ActionSelection')
        return selected

    def close(self):
        close = getattr(self.runtime, 'close', None)
        if close is not None:
            close()
