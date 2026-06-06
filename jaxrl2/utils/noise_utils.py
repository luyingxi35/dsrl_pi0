from typing import Sequence, Tuple

import numpy as np


def make_full_horizon_noise(actions_noise, action_chunk_shape: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return sequence noise and batched pi0 noise without horizon padding."""
    expected_shape = tuple(int(dim) for dim in action_chunk_shape)
    expected_size = int(np.prod(expected_shape))
    actions_noise = np.asarray(actions_noise, dtype=np.float32)

    if actions_noise.shape == expected_shape:
        sequence_noise = actions_noise
    elif actions_noise.size == expected_size:
        sequence_noise = actions_noise.reshape(expected_shape)
    else:
        raise ValueError(
            f"Expected a single noise sample with shape {expected_shape} "
            f"or total size {expected_size}, "
            f"got {actions_noise.shape}."
        )

    return sequence_noise, sequence_noise[np.newaxis]
