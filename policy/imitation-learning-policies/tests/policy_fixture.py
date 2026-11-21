from pathlib import Path
from types import SimpleNamespace
import gc
import unittest

import hydra
from omegaconf import OmegaConf
import torch
from torch import nn

from imitation_learning.models.encoders.image_encoders import SharedModelManager
from imitation_learning.utils.config_utils import compose_hydra_config
from robot_utils.config_utils import register_resolvers


CONFIG_DIR = Path(__file__).resolve().parents[1] / "imitation_learning/configs"
MODEL_NAME = "facebook/dinov2-with-registers-base"


class DummyDinoV2(nn.Module):
    """Exercise the actual encoders without downloading backbone weights."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="dinov2_with_registers",
            hidden_size=8,
            patch_size=14,
            num_register_tokens=4,
        )
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        positions = torch.arange(261, dtype=pixel_values.dtype)[None, :, None]
        channels = torch.arange(8, dtype=pixel_values.dtype)[None, None, :]
        content = pixel_values.mean(dim=(1, 2, 3)).reshape(batch_size, 1, 1)
        return SimpleNamespace(last_hidden_state=content + (positions + channels).sin())


class PolicyFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not OmegaConf.has_resolver("eval"):
            register_resolvers()

    def tearDown(self):
        SharedModelManager.reset()

    def make_config(self, cached=True, merge_size=4):
        with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            cfg = compose_hydra_config("rmbench", "diffusion_memory_transformer")
        cfg.device = "cpu"
        cfg.visual_encoder.frozen = True
        cfg.visual_encoder.spatial_merge_size = merge_size
        # Tests choose merger modes explicitly, independently of task defaults.
        cfg.visual_encoder.state_query_merge = False
        cfg.workspace.train_dataset.use_cached_image_features = cached
        model = cfg.workspace.model
        for name in (
            "global_cond_encoder", "local_cond_encoder",
            "history_img_feature_encoder",
        ):
            model[name].feature_dim = 8
        model.denoising_network_partial.hidden_dim = 32
        model.denoising_network_partial.head_num = 4
        model.denoising_network_partial.layer_num = 1
        model.denoising_network_partial.max_history_len = 2
        # These existing cases cover legacy checkpoints. The state-token route
        # is exercised explicitly in test_state_dit_tokens.py.
        model.denoising_network_partial.state_as_dit_token = False
        # Shadow lifecycle is covered separately in test_shadow_memory.
        model.denoising_network_partial.shadow_horizon = 0
        model.memory_eviction.keep_recent = 0
        model.noise_scheduler.inference_step_num = 1
        return cfg

    def instantiate_policy(self, cfg):
        # Policy destructors reset the singleton; collect any previous graph
        # cycles before registering the next test's backbone.
        gc.collect()
        SharedModelManager.reset()
        manager = SharedModelManager(cache_size=10)
        manager.register_vit_model(MODEL_NAME, frozen=True, model=DummyDinoV2())
        return hydra.utils.instantiate(cfg.workspace.model)

    def make_batch(self, cached=True, anchors=3):
        torch.manual_seed(2)
        batch = {
            "agent_pos": torch.randn(1, anchors, 1, 14),
            "action": torch.randn(1, anchors, 16, 14),
            "entire_traj_is_padding": torch.zeros(1, anchors, dtype=torch.bool),
            "traj_idx": (10 * torch.arange(anchors)).reshape(1, anchors, 1),
        }
        if cached:
            batch["head_camera_feature"] = torch.randn(1, anchors, 1, 257, 8)
        else:
            batch["head_camera"] = torch.rand(1, anchors, 1, 3, 224, 224)
        return batch

