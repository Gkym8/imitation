import click
import dill
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from robot_utils.torch_utils import torch_load
from imitation_learning.workspaces.base_workspace import BaseWorkspace


@click.command()
@click.option("--ckpt_path", type=str, required=True)
@click.option(
    "--record_history_attention",
    is_flag=True,
    help="Save offline frame/slot attention statistics once per epoch.",
)
@click.option(
    "--train_batch_size",
    type=click.IntRange(min=1),
    default=None,
    help="Optional per-process training batch size override.",
)
def main(
    ckpt_path: str,
    record_history_attention: bool,
    train_batch_size: int | None,
):
    ckpt = torch_load(ckpt_path, pickle_module=dill)

    config_str: str = ckpt["cfg_str_unresolved"]
    config = OmegaConf.create(config_str)
    OmegaConf.set_struct(config, True)
    assert type(config) == DictConfig
    # Update some configs here
    # config["workspace"]["train_dataset"]["dataloader_cfg"]["batch_size"] = 128

    if record_history_attention:
        denoising_cfg = config["workspace"]["model"][
            "denoising_network_partial"
        ]
        with open_dict(denoising_cfg):
            denoising_cfg["record_history_attention_epoch_stats"] = True
        print("Offline history-attention epoch statistics are enabled")

    if train_batch_size is not None:
        config["workspace"]["train_dataset"]["dataloader_cfg"][
            "batch_size"
        ] = train_batch_size
        print(f"Per-process training batch size: {train_batch_size}")

    config["workspace"]["cfg_str_unresolved"] = config_str
    workspace: BaseWorkspace = hydra.utils.instantiate(config["workspace"])
    workspace.resume_training(ckpt)


if __name__ == "__main__":
    main()
