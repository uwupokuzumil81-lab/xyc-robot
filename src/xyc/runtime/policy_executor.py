"""Map selected model actions to one unified platform command sink."""

from dataclasses import dataclass
from typing import Protocol, Sequence

from xyc.learning.contracts import robot_action


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

    def dispatch(self, command: dict, *, sequence: int, valid_for_s: float) -> CommandResult: ...

    def stop(self): ...


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

    def reset(self):
        self.runner.reset()
        self.sequence = 0

    def step(self, batch, *, observation=None, video_frames=None):
        action = self.runner.step(batch, observation=observation, video_frames=video_frames)
        values = action[0].detach().cpu().tolist()
        command = robot_action(values, self.runner.adapter.schema)
        mobile = command['mobile_command']
        submitted = [*command['arm_position_normalized'],
                     mobile['lift'], mobile['x'], mobile['y'], mobile['yaw']]
        self.sequence += 1
        try:
            result = self.command_sink.dispatch(
                command, sequence=self.sequence, valid_for_s=command['action_period_s'])
        except BaseException as exc:
            self.runner.record_execution(
                sent_action=submitted, accepted=False, command_id=str(self.sequence),
                message=f'{type(exc).__name__}: {exc}')
            try:
                self.command_sink.stop()
            except BaseException:
                pass
            raise
        if not isinstance(result, CommandResult):
            raise TypeError('Command sink must return CommandResult')
        self.runner.record_execution(
            sent_action=submitted, accepted=result.accepted, command_id=result.command_id,
            source_time_ns=result.source_time_ns, source_clock=result.source_clock,
            message=result.message)
        if result.feedback_state is not None:
            self.runner.record_feedback(
                state=result.feedback_state, source_time_ns=result.feedback_time_ns,
                source_clock=result.feedback_clock, message=result.message)
        return result

    def stop(self):
        self.command_sink.stop()

    def close(self):
        try:
            self.command_sink.stop()
        finally:
            self.runner.close()
