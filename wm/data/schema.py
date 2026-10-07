"""Shared constants: actions, dataset sources, end reasons, and the HDF5 schema (ROADMAP 4.3, 4.4)."""

import numpy as np

# Actions (4.3). NULL_ACTION is the classifier-free guidance token and never appears in data.
ACTION_MEANINGS = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
NUM_ACTIONS = 6
NULL_ACTION = 6
LAST_ROW_ACTION = -1  # stored action on the last row of every episode

# episodes/source
SOURCE_AGENT = 0
SOURCE_RANDOM = 1
SOURCE_HUMAN = 2

# episodes/end_reason
END_TERMINATED = 1
END_TRUNCATED = 2
END_STOPPED = 3  # human quit

SCHEMA_VERSION = 1
RAM_SIZE = 128

# Seed ranges (4.4, CP2-A, CP3-A). Training seeds stay below EVAL_SEED_BASE.
EPISODE_UID_STRIDE = 10_000_000  # episode_uid = worker * stride + local_counter
EVAL_SEED_BASE = 1_000_000_000
HUMAN_SEED_BASE = 2_000_000_000

# Per-row datasets: name -> (dtype, trailing shape). The frames shape is (H, W, 3) at the
# dataset's resolution, so it is None here and comes from ShardWriter's frame_shape.
ROW_FIELDS: dict[str, tuple[np.dtype, tuple[int, ...] | None]] = {
    "frames": (np.dtype(np.uint8), None),
    "actions": (np.dtype(np.int8), ()),
    "rewards": (np.dtype(np.float32), ()),
    "lives": (np.dtype(np.int8), ()),
    "ram": (np.dtype(np.uint8), (RAM_SIZE,)),
    "episode_idx": (np.dtype(np.int32), ()),
    "step_in_episode": (np.dtype(np.int32), ()),
    "terminal": (np.dtype(np.bool_), ()),
}

# Per-episode datasets, all shape (E,), in group EPISODES_GROUP.
EPISODES_GROUP = "episodes"
EPISODE_FIELDS: dict[str, np.dtype] = {
    "start": np.dtype(np.int64),
    "length": np.dtype(np.int64),  # rows = T + 1
    "episode_uid": np.dtype(np.int64),
    "seed": np.dtype(np.int64),
    "noops": np.dtype(np.int16),
    "source": np.dtype(np.int8),
    "epsilon": np.dtype(np.float32),  # -1 if not applicable
    "end_reason": np.dtype(np.int8),
    "return_": np.dtype(np.float32),
}

FILE_ATTRS = (
    "schema_version", "env_id", "frameskip", "maxpool", "resolution", "resize_method",
    "ale_py_version", "gymnasium_version", "git_commit", "created_utc", "collector_config",
)

FRAME_CHUNK_ROWS = 32
FRAME_COMPRESSION = "lzf"
FLUSH_EVERY = 1024
MAX_ROWS = 100_000
