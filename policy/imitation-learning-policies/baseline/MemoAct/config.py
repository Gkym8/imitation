from omegaconf import DictConfig

from baseline.Fusion.config import configure_fusion


def configure_memoact(cfg: DictConfig) -> None:
    configure_fusion(cfg)
    model = cfg.workspace.model
    model.memory_eviction = None
    model.denoising_network_partial.max_history_len = (
        model.memoact_short_term_capacity + model.memoact_long_term_capacity
    )
