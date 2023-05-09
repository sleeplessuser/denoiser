import yaml

import numpy as np


def read_config(path: str):
    with open(path, "r") as f:
        result = yaml.safe_load(f)
    return result


def int16_to_fp32(data: np.ndarray) -> np.ndarray:
    assert data.dtype == np.int32, f"int16 array expected, got {data.dtype}"
    i = np.iinfo(np.int32)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (data.astype(np.float32) - offset) / abs_max


def fp32_to_int16(data: np.ndarray) -> np.ndarray:
    assert data.dtype == np.float32, f"float32 array expected, got {data.dtype}"
    i = np.iinfo(np.int16)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (data * abs_max + offset).clip(i.min, i.max).astype(np.int16)
