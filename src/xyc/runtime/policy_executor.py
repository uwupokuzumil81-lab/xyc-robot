"""Map selected model actions to one unified platform command sink."""

from dataclasses import dataclass
from typing import Protocol, Sequence
import time

from xyc.contracts import COMMAND_VALIDITY_S, robot_action


@dataclass(frozen=True)
class CommandResult:
    """What the command transport actually knows about one dispatch."""

    accepted: bool | None = None
    command_id: str | None = None
    source_time_ns: int | None = None
    source_clock: str | None = None
    message: str = ''
    feedback_state: Sequence[float] | None = None
    feedback_time_ns: int | None = None
    feedback_clock: str | None = None


class CommandSink(Protocol):
    """ROS, replay transport or another frontend can implement this contract."""

    def dispatch(self, command: dict, *, sequence: int, valid_for_s: float,
                 before_send=None) -> CommandResult | None: ...

    def stop(self): ...

    def fault_stop(self): ...


class PolicyExecutor:
    """Run a policy tick and dispatch the same 18-D platform contract.

    This class deliberately contains no ROS, serial, motor limits or RTC math.
    A command sink routes the named command into the existing unified robot
    controller; all hardware protection remains below that boundary.
    """

    def __init__(self, runner, command_sink: CommandSink):
        self.runner = runner
        self.command_sink = command_sink
        self.sequence = 0

    def step(self, batch, *, observation=None, video_frames=None, deadline=None, allow_dispatch=None):
        action = self.runner.step(batch, observation=observation, video_frames=video_frames)
        if action is None:
            return None
        action_deadline = self.runner.action_deadline
        if action_deadline is not None:
            deadline = action_deadline if deadline is None else min(deadline, action_deadline)
        values = action[0].detach().cpu().tolist()
        return self.execute_action(values, deadline=deadline, observation=observation, before_send=allow_dispatch)

    def execute_action(self, values, *, record=True, deadline=None, observation=None, before_send=None):
        """Manual preparation and model execution share one command path.

        Regression outputs are projected onto the full calibrated position
        domain. Raw predictions and the actual bounded command remain separate
        trace records; there is no step-size or target-ahead restriction.
        """
        command = robot_action(values, self.runner.adapter.schema, bound_arms=True)
        if observation is not None and observation.get('source_clock') == 'robot_ros':
            command['observation_time_ns'] = observation['source_time_ns']
        mobile = command['mobile_command']
        submitted = [*command['arm_position_normalized'],
                     mobile['lift'], mobile['x'], mobile['y'], mobile['yaw']]
        self.sequence += 1
        validity = (COMMAND_VALIDITY_S if deadline is None else
                    min(COMMAND_VALIDITY_S, deadline - time.monotonic()))
        if validity <= 0:
            if record:
                self.runner.record_execution(
                    sent_action=None, accepted=False, command_id=str(self.sequence),
                    message='Inference deadline expired before dispatch; no action was sent')
            raise TimeoutError('Inference deadline expired before dispatch; no action was sent')
        try:
            result = self.command_sink.dispatch(
                command, sequence=self.sequence, valid_for_s=validity, before_send=before_send)
        except BaseException as exc:
            if record:
                self.runner.record_execution(
                    sent_action=submitted, accepted=None, command_id=str(self.sequence),
                    message=f'{type(exc).__name__}: {exc}')
            raise
        if result is None:
            if record:
                self.runner.record_execution(sent_action=None, accepted=False, command_id=str(self.sequence),
                                             message='Model action cancelled before TCP send')
            return None
        if not isinstance(result, CommandResult):
            raise TypeError('Command sink must return CommandResult')
        if record:
            self.runner.record_execution(
                sent_action=submitted, accepted=result.accepted, command_id=result.command_id,
                source_time_ns=result.source_time_ns, source_clock=result.source_clock,
                message=result.message)
        if record and result.feedback_state is not None:
            self.runner.record_feedback(
                state=result.feedback_state, source_time_ns=result.feedback_time_ns,
                source_clock=result.feedback_clock, message=result.message)
        return result

    def close(self, *, fault=False):
        try:
            if fault:
                self.command_sink.fault_stop()
            else:
                self.command_sink.stop()
        finally:
            self.runner.close()
