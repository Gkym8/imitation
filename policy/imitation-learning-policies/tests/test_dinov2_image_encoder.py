import types
import unittest

import torch
from torch import nn

from imitation_learning.models.encoders.image_encoders import (
    SharedDinoV2ImageEncoder,
    SharedModelManager,
)


MODEL_NAME = "facebook/dinov2-with-registers-base-test"


class DummyDinoV2(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(
            model_type="dinov2_with_registers",
            hidden_size=hidden_size,
            patch_size=14,
            num_register_tokens=4,
        )
        self.anchor = nn.Parameter(torch.zeros(()))
        self.last_input: torch.Tensor | None = None

    def forward(self, pixel_values: torch.Tensor):
        self.last_input = pixel_values.detach().clone()
        patch_num = (pixel_values.shape[-2] // 14) * (
            pixel_values.shape[-1] // 14
        )
        token_num = 1 + self.config.num_register_tokens + patch_num
        values = torch.arange(
            token_num,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )[None, :, None]
        output = values.expand(pixel_values.shape[0], -1, self.config.hidden_size)
        return types.SimpleNamespace(last_hidden_state=output)


def make_meta(size: int = 256) -> dict:
    return {
        "name": "head_camera",
        "shape": [3, size, size],
        "data_type": "image",
        "length": 1,
        "normalizer": "identity",
        "augmentation": [],
        "source_entry_names": ["head_camera"],
    }


class SharedDinoV2ImageEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        SharedModelManager.reset()
        self.manager = SharedModelManager(cache_size=10)
        self.model = DummyDinoV2()
        self.manager.register_vit_model(
            MODEL_NAME, frozen=True, model=self.model
        )

    def tearDown(self) -> None:
        SharedModelManager.reset()

    def make_encoder(self, aggregation: str) -> SharedDinoV2ImageEncoder:
        return SharedDinoV2ImageEncoder(
            feature_aggregation=aggregation,
            apply_image_norm=True,
            resize_size=224,
            individual_forward=False,
            model_name=MODEL_NAME,
            image_meta=make_meta(),
            pretrained=True,
            frozen=True,
        )

    def test_resize_normalize_and_discard_special_tokens(self) -> None:
        encoder = self.make_encoder("patches")
        output = encoder(torch.ones(2, 1, 3, 256, 256))

        self.assertEqual(output.shape, (2, 1, 256, 8))
        # CLS is token 0 and registers are tokens 1..4; patch 0 is token 5.
        torch.testing.assert_close(output[0, 0, 0], torch.full((8,), 5.0))
        assert self.model.last_input is not None
        self.assertEqual(self.model.last_input.shape, (2, 3, 224, 224))
        expected_channel_zero = (1.0 - 0.485) / 0.229
        torch.testing.assert_close(
            self.model.last_input[0, 0, 0, 0],
            torch.tensor(expected_channel_zero),
        )

    def test_map_alias_returns_one_cls_token_for_six_dimensional_input(self) -> None:
        encoder = self.make_encoder("map")
        output = encoder(torch.zeros(2, 3, 1, 3, 256, 256))

        self.assertEqual(output.shape, (2, 3, 1, 1, 8))
        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_individual_forward_processes_the_complete_flattened_batch(self) -> None:
        encoder = self.make_encoder("patches")
        encoder.individual_forward = True
        output = encoder(torch.zeros(2, 3, 1, 3, 256, 256))

        self.assertEqual(output.shape, (2, 3, 1, 256, 8))

    def test_rejects_resize_not_divisible_by_patch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by patch_size"):
            SharedDinoV2ImageEncoder(
                feature_aggregation="patches",
                apply_image_norm=True,
                resize_size=225,
                model_name=MODEL_NAME,
                image_meta=make_meta(),
                pretrained=True,
                frozen=True,
            )


if __name__ == "__main__":
    unittest.main()
