from omegaconf import DictConfig


def configure_fusion(cfg: DictConfig) -> None:
    """Apply the Fusion ablations after merging task and launch overrides."""
    model = cfg.workspace.model
    model.action_history = False
    if "memory_gate" in model:
        model.memory_gate = None
    model.denoising_network_partial.use_conditioned_slot_gate = False
    model.denoising_network_partial.shadow_horizon = 0
    cfg.workspace.trainer.gate_selectivity_loss_weight = 0.0
