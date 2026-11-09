import hydra
from hydra.errors import MissingConfigException
from omegaconf import DictConfig, OmegaConf
from typing import cast


def remove_keys_from_config(cfg: DictConfig):
    """
    Will read _keys_to_remove_ cfg and remove the keys from the config
    This function should be called right before instantiating the config
    """
    if "_keys_to_remove_" not in cfg:
        return cfg
    keys_to_remove = cfg["_keys_to_remove_"]
    is_struct = OmegaConf.is_struct(cfg)
    if is_struct:
        OmegaConf.set_struct(cfg, False)  # Make cfg mutable
    OmegaConf.resolve(cfg)
    for key in keys_to_remove:
        parent = OmegaConf.select(cfg, ".".join(key.split(".")[:-1]))
        if parent is not None:
            print(f"Removing {key} from cfg")
            parent.pop(key.split(".")[-1])
    if is_struct:
        OmegaConf.set_struct(cfg, True)  # Make cfg immutable to ensure proper override

    return cfg

def _compose_task_config(task_name: str, fallback_task_name: str | None):
    def load(name: str) -> DictConfig:
        task_cfg = cast(DictConfig, hydra.compose(f"task/{name}.yaml")["task"])
        OmegaConf.set_struct(task_cfg, False)
        return task_cfg

    if fallback_task_name is None:
        return load(task_name), task_name

    fallback_cfg = load(fallback_task_name)
    if task_name == fallback_task_name:
        return fallback_cfg, fallback_task_name
    try:
        task_cfg = load(task_name)
    except MissingConfigException as error:
        # Only a missing task file falls back. Broken defaults in an existing
        # task file must still report the configuration error.
        if error.missing_cfg_file != f"task/{task_name}.yaml":
            raise
        return fallback_cfg, fallback_task_name
    return cast(DictConfig, OmegaConf.merge(fallback_cfg, task_cfg)), task_name


def compose_hydra_config(
    task_name: str, policy_name: str, fallback_task_name: str | None = None,
):
    """
    Config override order:
    - base_config.yaml
    - workspace/policy/{policy_name}.yaml
    - workspace/dataset/{dataset_name}.yaml
    - task/{fallback_task_name}.yaml (optional defaults)
    - task/{task_name}.yaml (if present when a fallback is supplied)
    Outside this function: train_policy.yaml (for customized overrides)
    """


    base_cfg = hydra.compose(f"base_config.yaml")
    OmegaConf.set_struct(base_cfg, False)  # Make cfg mutable
    policy_cfg = hydra.compose(f"workspace/policy/{policy_name}.yaml")
    OmegaConf.set_struct(policy_cfg, False)  # Make cfg mutable
    policy_cfg["workspace"]["model"] = policy_cfg["workspace"].pop("policy") # Replace the name to policy

    base_cfg = cast(DictConfig, OmegaConf.merge(base_cfg, policy_cfg))

    task_cfg, selected_task_name = _compose_task_config(task_name, fallback_task_name)
    OmegaConf.set_struct(task_cfg, False)  # Make cfg mutable
    dataset_type = task_cfg["dataset_type"]
    """
    Valid dataset types:
    - mujoco
    - robomimic
    - iphumi
    - iphumi_multi_task
    - real_world
    - umi
    - umi_multi_task
    - mikasa
    """

    if "memory" in policy_name.lower() or "gated" in policy_name.lower():
        dataset_cfg_name = f"{dataset_type}_multi_traj"
    else:
        dataset_cfg_name = f"{dataset_type}_single_traj"

    # Include base dataset config
    print(f"Using dataset config: {dataset_cfg_name}")
    dataset_cfg = hydra.compose(f"workspace/dataset/{dataset_cfg_name}.yaml")
    OmegaConf.set_struct(dataset_cfg, False)  # Make cfg mutable
    dataset_cfg["workspace"]["train_dataset"] = dataset_cfg["workspace"].pop("dataset")

    base_cfg = cast(DictConfig, OmegaConf.merge(base_cfg, dataset_cfg))
    
    # Apply task overrides
    base_cfg = cast(DictConfig, OmegaConf.merge(base_cfg, task_cfg))
    # base_cfg["task_name"] = task_name
    base_cfg["task_name"] = task_name
    base_cfg["policy_name"] = policy_name
    if fallback_task_name is not None:
        base_cfg["task_yaml_name"] = selected_task_name

    OmegaConf.set_struct(base_cfg, True)  # Make cfg immutable to ensure proper override

    return base_cfg


def compose_training_config(cfg: DictConfig) -> DictConfig:
    """Compose task defaults, optional task overrides, then launch overrides."""
    task_yaml_name = cfg.get("task_yaml_name", cfg["task_name"])
    # The RMBench launcher opts into inheritance; other launchers keep their
    # existing task-file selection unless they also specify a fallback.
    fallback = cfg.get("task_yaml_fallback")
    base_cfg = compose_hydra_config(task_yaml_name, cfg["policy_name"], fallback)
    selected = base_cfg.get("task_yaml_name", task_yaml_name)
    OmegaConf.set_struct(base_cfg, False)
    base_cfg["task_yaml_name"] = selected
    if "task_yaml_fallback" in cfg:
        base_cfg["task_yaml_fallback"] = cfg["task_yaml_fallback"]
    OmegaConf.set_struct(base_cfg, True)
    merged = cast(DictConfig, OmegaConf.merge(base_cfg, cfg))
    merged["task_yaml_name"] = selected
    description = f"task/{selected}.yaml"
    if fallback is not None and selected != fallback:
        description += f" (overrides task/{fallback}.yaml)"
    print(f"Using task config: {description}")
    return merged


def compose_memory_gate_hydra_config(task_name: str):
    base_cfg = hydra.compose(f"memory_gate_config.yaml")
    OmegaConf.set_struct(base_cfg, False)  # Make cfg mutable

    # Apply task overrides
    task_cfg = hydra.compose(f"task/{task_name}.yaml")
    assert isinstance(task_cfg, DictConfig)
    task_cfg = task_cfg["task"]

    dataset_type = task_cfg["dataset_type"]
    """
    Valid dataset types:
    - mujoco
    - robomimic
    - iphumi
    - iphumi_multi_task
    - real_world
    - umi
    - umi_multi_task
    """

    dataset_cfg_name = f"{dataset_type}_single_traj"

    # Include base dataset config
    print(f"Using dataset config: {dataset_cfg_name}")
    dataset_cfg = hydra.compose(f"workspace/dataset/{dataset_cfg_name}.yaml")
    OmegaConf.set_struct(dataset_cfg, False)  # Make cfg mutable
    dataset_cfg["workspace"]["train_dataset"] = dataset_cfg["workspace"].pop("dataset")
    base_cfg = cast(DictConfig, OmegaConf.merge(base_cfg, dataset_cfg))
    
    base_cfg = cast(DictConfig, OmegaConf.merge(base_cfg, task_cfg))
    # base_cfg["task_name"] = task_name
    base_cfg["task_name"] = task_name

    OmegaConf.set_struct(base_cfg, True)  # Make cfg immutable to ensure proper override

    
    return base_cfg
