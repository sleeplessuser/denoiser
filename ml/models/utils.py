from torch import nn

from .demucs import Demucs
from .wavenet import Wavenet

def get_model(name: str = 'demucs64') -> nn.Module:
    name = name.lower()
    if name == 'demucs48':
        return Demucs(hidden=48)
    elif name == 'demucs64':
        return Demucs(hidden=64)
    elif name == 'wavenet':
        return Wavenet()
    else:
        raise ValueError(f'Unknown model name: {name}')