from pathlib import Path
from typing import Callable

import torch
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio


def pad_batch(batch: list[torch.Tensor]) -> list[torch.Tensor]:
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    tensors, targets = [], []
    for x, y in batch:
        tensors += [x]
        targets += [y]

    tensors = pad_batch(tensors)
    targets = pad_batch(targets)

    return tensors, targets


def noisy_name_mapping(name: str, clean_path: str) -> str:
    filename = Path(name).name
    clean_name = filename.split("_")[-1]
    return str(Path(clean_path) / clean_name)


class AudioDataset(Dataset):
    def __init__(
        self,
        clean_data_dir: str,
        noisy_data_dir: str,
        noise_name_mapping: Callable[[str, str], str],
        sample_rate: int = 16000,
    ) -> None:
        self.clean_data_dir = clean_data_dir
        self.noisy_data_dir = noisy_data_dir
        self.noisy_filenames = sorted(Path(noisy_data_dir).glob("*"))
        self.name_mapping = noise_name_mapping
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.noisy_filenames)

    def __getitem__(self, index: int):
        noisy_name = self.noisy_filenames[index]
        clean_name = self.name_mapping(noisy_name, self.clean_data_dir)

        noisy, noisy_sr = self.load_audio(noisy_name)
        clean, clean_sr = self.load_audio(clean_name)

        noisy = self.prepare_wav(noisy, noisy_sr)
        clean = self.prepare_wav(clean, clean_sr)

        return noisy, clean

    def prepare_wav(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if wav.shape[0] > 1:  # convert to mono
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.sample_rate)
        return wav

    @staticmethod
    def load_audio(filename: str):
        return torchaudio.load(filename)


class AudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        clean_data_dir: str,
        noisy_data_dir: str,
        noise_name_mapping: Callable[[str, str], str],
        sample_rate: int = 16000,
        batch_size=64,
        num_workers=20,
        ratio=(0.9, 0.1),
        random_seed=42,
    ) -> None:
        super().__init__()
        self.clean_data_dir = clean_data_dir
        self.noisy_data_dir = noisy_data_dir
        self.name_mapping = noise_name_mapping
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ratio = ratio
        self.random_seed = random_seed

    def setup(self, stage: str) -> None:
        dataset = AudioDataset(
            self.clean_data_dir,
            self.noisy_data_dir,
            self.name_mapping,
            self.sample_rate,
        )
        generator = torch.Generator().manual_seed(self.random_seed)
        train_size = int(len(dataset) * self.ratio[0])
        val_size = len(dataset) - train_size

        self.train_workers = int(self.num_workers * 0.6)
        self.val_workers = max((self.num_workers - self.train_workers) // 2, 1)
        self.test_workers = self.val_workers

        self.train_data, self.val_data = random_split(
            dataset, [train_size, val_size], generator
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=True,
            num_workers=self.train_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=True,
            num_workers=self.val_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=True,
            num_workers=self.test_workers,
            pin_memory=True,
        )
