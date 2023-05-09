from pathlib import Path
import argparse

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from ml.data import AudioDataModule, noisy_name_mapping
from ml.models.wrapper import LitWrapper
from ml.loss import get_loss
from ml.utils import read_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="training config path", default="ml/configs/demucs64.yaml"
    )
    
    parser.add_argument(
        "--name", help="experiment name", default="demucs64"
    )

    parser.add_argument(
        "--wandb",
        help="log metrics and models to weights and biases",
        default=False,
        action="store_true",
    )
    
    parser.add_argument(
        "--noisy",
        help="noisy data path",
        required=True
    )

    parser.add_argument(
        "--clean",
        help="clean data path",
        required=True
    )

    parser.add_argument(
        "--save-dir", help="directory to save results", default="ml/save_dir"
    )

    parser.add_argument(
        "--device", help="either cpu or gpu", default="gpu", choices=["cpu", "gpu"]
    )

    parser.add_argument(
        "--precision", help="training precision", default=16, choices=[16, 32]
    )

    return parser.parse_args()


def main(args):
    config = read_config(args.config)
    pl.seed_everything(config["train"]["seed"])

    loss_fn = get_loss()
    model = LitWrapper(config=config, loss_fn=loss_fn)
    save_dir = Path(args.save_dir)
    logger = None

    if args.wandb:
        logger = WandbLogger(
            name=args.name,
            project="denoiser",
            log_model=True,
            save_dir=save_dir,
        )
    else:
        logger = TensorBoardLogger(save_dir=save_dir / args.name, name=None)

    data_module = AudioDataModule(
        clean_data_dir=args.clean,
        noisy_data_dir=args.noisy,
        noise_name_mapping=noisy_name_mapping,
        sample_rate=config["model"]["sr"],
        batch_size=config["train"]["batch"],
        num_workers=config["train"]["workers"],
        ratio=config["train"]["ratio"],
        random_seed=config["train"]["seed"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir / "checkpoints" / args.name,
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        save_weights_only=True,
    )

 
    trainer = pl.Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator=args.device, 
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        precision=args.precision,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path='best')
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
