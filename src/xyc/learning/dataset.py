"""PyTorch action-window dataset, using the shared v3 metadata/video reader."""

from bisect import bisect_right
import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import ACTION, OBS_STATE
from .dataset_metadata import DatasetMetadata, Episode, RGBVideos


class EpisodeDataset(Dataset):
    def __init__(self, metadata, episodes, *, chunk_size, image_size):
        self.metadata, self.episodes = metadata, list(episodes)
        self.chunk_size, self.image_size = int(chunk_size), tuple(image_size)
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
        indices = np.minimum(np.arange(frame, frame + self.chunk_size), length - 1)
        result = {OBS_STATE: torch.from_numpy(numeric[OBS_STATE][frame].copy()),
                  ACTION: torch.from_numpy(numeric[ACTION][indices].copy()),
                  'action_is_pad': torch.arange(frame, frame + self.chunk_size) >= length}
        if self.reader is None:
            self.reader = RGBVideos(self.image_size)
        for camera in self.metadata.cameras:
            path, offset = self.metadata.video(episode, camera)
            image = self.reader.read(path, offset + frame / self.metadata.schema.fps, self.metadata.schema.fps)
            result['observation.images.' + camera] = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        return result

    def __getstate__(self):
        state = dict(self.__dict__)
        state['reader'] = None
        return state

    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None
