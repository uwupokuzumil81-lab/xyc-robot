"""PyTorch action-window dataset, using the shared v3 metadata/video reader."""

from bisect import bisect_right
import numpy as np
import torch
from torch.utils.data import Dataset

from xyc.contracts import ACTION, OBS_IMAGES, OBS_STATE
from xyc.images import decode_jpeg
from .dataset_metadata import RGBVideos


def live_observation_batch(observation, schema):
    """Produce the same raw one-sample RGB/state/task batch as offline data."""
    state, images = observation.get('state'), observation.get('images')
    if (not isinstance(state, (tuple, list)) or len(state) != len(schema.state_names)
            or not isinstance(images, dict) or set(images) != set(schema.cameras)):
        raise ValueError('Live observation does not match the checkpoint schema')
    batch = {OBS_STATE: torch.tensor([state], dtype=torch.float32)}
    for name in schema.cameras:
        rgb = decode_jpeg(images[name])
        batch[f'{OBS_IMAGES}.{name}'] = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    if 'task' in observation:
        batch['task'] = [observation['task']]
    return batch


class EpisodeDataset(Dataset):
    def __init__(self, metadata, episodes, *, spec):
        self.metadata, self.episodes = metadata, list(episodes)
        self.spec = spec
        self.ends = np.cumsum([metadata.episodes[i].length for i in self.episodes]).tolist()
        self.reader = None

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        position = bisect_right(self.ends, index)
        return self.episodes[position], index - (self.ends[position - 1] if position else 0)

    def __getitem__(self, index):
        episode, frame = self.locate(int(index))
        numeric = self.metadata.numeric(episode)
        length = len(numeric[ACTION])
        def indices(offsets):
            requested = frame + np.asarray(offsets)
            return np.clip(requested, 0, length - 1), torch.from_numpy((requested < 0) | (requested >= length))

        action_indices, padding = indices(self.spec.action_offsets)
        result = {ACTION: torch.from_numpy(numeric[ACTION][action_indices].copy()), 'action_is_pad': padding}
        if self.spec.state_offsets is None:
            result[OBS_STATE] = torch.from_numpy(numeric[OBS_STATE][frame].copy())
        else:
            state_indices, padding = indices(self.spec.state_offsets)
            result[OBS_STATE] = torch.from_numpy(numeric[OBS_STATE][state_indices].copy())
            result[OBS_STATE + '_is_pad'] = padding
            result['observation.current_state'] = torch.from_numpy(numeric[OBS_STATE][frame].copy())
        if self.reader is None:
            self.reader = RGBVideos()
        image_indices, padding = indices(self.spec.image_offsets or (0,))
        for camera in self.metadata.cameras:
            path, offset = self.metadata.video(episode, camera)
            images = [torch.from_numpy(self.reader.read(
                path, offset + int(i) / self.metadata.schema.fps, self.metadata.schema.fps))
                .permute(2, 0, 1).contiguous() for i in image_indices]
            key = f'{OBS_IMAGES}.{camera}'
            result[key] = images[0] if self.spec.image_offsets is None else torch.stack(images)
            if self.spec.image_offsets is not None:
                result[key + '_is_pad'] = padding
                current = self.reader.read(path, offset + frame / self.metadata.schema.fps, self.metadata.schema.fps)
                result['observation.current_images.' + camera] = torch.from_numpy(current).permute(2, 0, 1).contiguous()
        if self.spec.include_task:
            result['task'] = self.metadata.task_text(episode, frame)
        return result

    def __getstate__(self):
        state = dict(self.__dict__)
        state['reader'] = None
        return state

    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None
