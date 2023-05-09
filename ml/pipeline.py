from typing import Tuple

import numpy as np
import torch
import torchaudio

from ml.models.wrapper import LitWrapper


class Pipeline:
    def __init__(
        self, model: LitWrapper, sample_rate: int = 16000, max_length_seconds: int = 10
    ) -> None:
        """
        Args:
            model (LitWrapper):
                denoiser model
            sample_rate (int, optional):
                model sample rate. Defaults to 16000.
            max_length_seconds (int, optional):
                max length of audio without splitting. Defaults to 10.
        """
        self.model = model
        self.sr = sample_rate
        self._warmup()

    def load_audio(self, path: str):
        wav, sr = torchaudio.load(path)
        return wav, sr

    def preprocess(self, wav: torch.Tensor, sr: int):
        if wav.shape[0] > 1:  # convert to mono
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav

    @torch.no_grad()
    def predict(self, wav: torch.Tensor):
        wav = wav.to(self.model.device)
        estimate = self.model(wav)
        return estimate

    def postprocess(self, estimate: torch.Tensor, sr: int):
        if sr != self.sr:
            estimate = torchaudio.functional.resample(estimate, self.sr, sr)
        return estimate

    def _warmup(self):
        dummy = torch.randn(1, 1, self.sr).to(self.model.device)
        self.predict(dummy)
        del dummy

    def denoise(self, filename: str) -> Tuple[int, np.ndarray]:
        """run denoising pipeline

        Args:
            filename (str): path to audio file

        Returns:
            Tuple[int, np.ndarray]: sample rate and denoised audio
        """
        wav, sr = self.load_audio(filename)
        wav = self.preprocess(wav, sr)
        estimate = self.predict(wav)
        estimate = self.postprocess(estimate, sr)
        estimate = estimate.cpu().detach().numpy()
        del wav
        return sr, estimate


def get_pipeline(ckpt: str, device="cuda") -> Pipeline:
    wrapper = LitWrapper.load_from_checkpoint(ckpt, map_location=device)
    wrapper = wrapper.eval().to(device)
    return Pipeline(wrapper, wrapper.config["model"]["sr"])
