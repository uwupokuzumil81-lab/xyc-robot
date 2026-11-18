"""One calibration identity for recording, replay, observations and deployment."""

from xyc.io import content_id


def follower_calibration_ids(assets):
    documents = assets.get('calibrations', {})
    ids = []
    for side in ('left', 'right'):
        document = documents.get('calibration/' + side)
        if not isinstance(document, dict) or document.get('arm') != side:
            raise ValueError('Missing follower calibration asset: ' + side)
        ids.append(content_id(document))
    return ids
