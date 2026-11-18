"""Raw RGB image decoding and explicit adapter-selected resizing."""

import cv2
import numpy as np


def decode_jpeg(encoded):
    if not isinstance(encoded, bytes) or not encoded.startswith(b'\xff\xd8'):
        raise ValueError('Expected JPEG image bytes')
    bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError('JPEG decode failed')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_rgb(image, size):
    """Stretch to (height,width); models requiring padding choose another transform."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected RGB uint8 HWC image')
    height, width = size
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
