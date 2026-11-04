import copy
from math import sqrt
from typing import Any, cast

import einops
import torch
from omegaconf import DictConfig
from torch import nn

from imitation_learning.common.dataclasses import DataMeta, construct_data_meta


class BaseImageEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        image_meta: DataMeta | dict[str, Any] | DictConfig,
        pretrained: bool,
        frozen: bool,
        **unused_kwargs,
    ):
        super().__init__()
        self.model_name: str = model_name
        self.image_meta: DataMeta = construct_data_meta(image_meta)
        shape_tuple = tuple(self.image_meta.shape)
        assert len(shape_tuple) == 3, f"shape_tuple dimension is not 3: {shape_tuple}"
        self.image_shape: tuple[int, int, int] = shape_tuple
        """(C, H, W)"""
        self.frozen: bool = frozen
        self.pretrained: bool = pretrained
        print(f"BaseImageEncoder unused_kwargs: {unused_kwargs}")
        self.feature_dim: int
        self.token_num: int
        self.model: nn.Module

    def forward(self, x: torch.Tensor):
        """
        x.shape: (batch_size, trajectory_len, C, H, W) or (batch_size, traj_num, traj_len, C, H, W)
        return: (batch_size, trajectory_len, token_num, feature_dim) or (batch_size, traj_num, traj_len, token_num, feature_dim)
        """
        raise NotImplementedError


class TimmImageEncoder(BaseImageEncoder):
    def __init__(
        self,
        model_name: str,
        image_meta: DataMeta | dict[str, Any],
        pretrained: bool,
        frozen: bool,
        feature_aggregation: str,
    ):
        super().__init__(
            model_name=model_name,
            image_meta=image_meta,
            pretrained=pretrained,
            frozen=frozen,
        )

        self.feature_aggregation: str = feature_aggregation
        from timm.models import create_model

        self.model = create_model(
            model_name=model_name, pretrained=pretrained, num_classes=0
        )

        if feature_aggregation == "cls" or feature_aggregation == "mean":
            self.token_num: int = 1
        elif feature_aggregation == "patches":
            self.token_num: int = self.model.patch_embed.num_patches
        else:
            raise ValueError(
                f"feature_aggregation: {feature_aggregation} is not supported"
            )
        print(
            f"TimmImageEncoder token_num: {self.token_num}, feature_aggregation: {feature_aggregation}, self.model.patch_embed.num_patches: {self.model.patch_embed.num_patches}"
        )

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

        self.feature_dim: int = self.model.embed_dim

    def forward(self, x: torch.Tensor):
        """
        x.shape: (batch_size, trajectory_len, C, H, W)
        return: (batch_size, trajectory_len, token_num, feature_dim)
        """

        batch_size, trajectory_len, *image_shape = x.shape
        assert (
            trajectory_len == self.image_meta.length
        ), f"trajectory_len: {trajectory_len}, self.image_meta.length: {self.image_meta.length} are not consistent"
        assert (
            tuple(image_shape) == self.image_shape
        ), f"image_shape: {tuple(image_shape)}, self.image_shape: {self.image_shape}"
        x = x.reshape(
            batch_size * trajectory_len, *image_shape
        )  # (batch_size * trajectory_len, C, H, W)

        feature = self.model.forward_features(
            x
        )  # (batch_size * trajectory_len, patch_num+1, feature_dim)
        assert (
            len(feature.shape) == 3
        ), f"feature.shape: {feature.shape}, feature.shape should be 3"
        patch_num = feature.shape[1] - 1
        if self.feature_aggregation == "cls":
            feature = feature[:, 0, :].reshape(
                batch_size, trajectory_len, 1, -1
            )  # (batch_size, trajectory_len, 1, feature_dim)
        elif self.feature_aggregation == "mean":
            feature = torch.mean(feature, dim=1).reshape(
                batch_size, trajectory_len, 1, -1
            )  # (batch_size, trajectory_len, 1, feature_dim)
        elif self.feature_aggregation == "patches":
            feature = feature[:, 1:, :].reshape(
                batch_size, trajectory_len, patch_num, -1
            )  # (batch_size, trajectory_len, patch_num, feature_dim)
        else:
            raise ValueError(
                f"feature_aggregation: {self.feature_aggregation} is not supported"
            )

        return feature


class CLIPImageEncoder(BaseImageEncoder):
    def __init__(
        self,
        model_name: str,
        image_meta: DataMeta | dict[str, Any],
        pretrained: bool,
        frozen: bool,
        feature_aggregation: str,
        use_lora: bool,
    ):
        super().__init__(
            model_name=model_name,
            image_meta=image_meta,
            pretrained=pretrained,
            frozen=frozen,
        )

        from transformers import (CLIPVisionConfig, CLIPVisionModel)

        self.model: nn.Module = CLIPVisionModel.from_pretrained(model_name)
        self.feature_dim: int = CLIPVisionConfig.from_pretrained(model_name).hidden_size

        # from transformers import AutoModel, AutoConfig

        # self.model: nn.Module = AutoModel.from_pretrained(model_name)
        # self.feature_dim: int = AutoConfig.from_pretrained(
        #     model_name
        # ).hidden_size

        print(f"Model name: {model_name}, feature dimension: {self.feature_dim}")

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

        assert feature_aggregation in [
            "cls",
            "mean",
        ], f"feature_aggregation: {feature_aggregation} is not supported"
        self.feature_aggregation: str = feature_aggregation

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"CLIPImageEncoder trainable parameters: {trainable_params}")

        if use_lora:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                # target_modules=["attention.query", "attention.value"],
                lora_dropout=0.1,
                inference_mode=False,
                bias="none",
            )
            # lora_config = LoraConfig(
            #     r=16,  # Rank of the LoRA matrices (smaller rank reduces trainable parameters)
            #     lora_alpha=32,  # Scaling factor for LoRA updates
            #     lora_dropout=0.1,  # Dropout rate applied to LoRA updates
            #     target_modules=["query", "key", "value"],  # Target the attention modules
            #     bias="none",  # Whether to train bias ("none", "all", or "lora_only")
            #     task_type="SEQ_CLS",  # Specify the task type (e.g., "SEQ_CLS" for sequence classification)
            # )
            self.model = get_peft_model(self.model, lora_config)  # TODO: fix bugs here

            for name, param in self.model.named_parameters():
                # Freeze all layers except LoRA-injected layers
                if "lora" not in name:
                    param.requires_grad = False

            trainable_params_lora = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print(
                f"CLIPImageEncoder trainable parameters after LoRA: {trainable_params_lora}, reduction rate: {(trainable_params - trainable_params_lora) / trainable_params}"
            )

    def forward(self, x: torch.Tensor):
        """x.shape: (B, T, C, H, W), Output: (B, T*D) where T is the number of frames and D is the feature dimension"""

        batch_size, obs_len, *image_shape = x.shape
        assert (
            obs_len == self.image_meta.length
        ), f"obs_len: {obs_len}, self.image_meta.length: {self.image_meta.length} are not consistent"
        assert (
            tuple(image_shape) == self.image_shape
        ), f"image_shape: {tuple(image_shape)}, self.image_shape: {self.image_shape}"

        x = x.reshape(batch_size * obs_len, *image_shape)

        # Ensure only pixel_values are passed
        outputs = self.model(pixel_values=x)

        if self.feature_aggregation == "cls":
            feature: torch.Tensor = outputs.last_hidden_state[:, 0, :]
        elif self.feature_aggregation == "mean":
            feature: torch.Tensor = outputs.last_hidden_state.mean(dim=1)
        else:
            raise ValueError(
                f"feature_aggregation: {self.feature_aggregation} is not supported"
            )

        return feature.reshape(batch_size, -1)


class SiglipImageEncoder(BaseImageEncoder):
    def __init__(
        self,
        feature_aggregation: str,
        apply_image_norm: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)

        from transformers import SiglipVisionModel

        self.model: nn.Module = SiglipVisionModel.from_pretrained(self.model_name)
        self.feature_dim: int = self.model.config.hidden_size

        # For some unknown reason, this does not work with autocast
        # self.processor: SiglipImageProcessor = SiglipImageProcessor.from_pretrained(self.model_name)
        # Manually assign the mean and std for siglip
        self.img_mean: nn.Parameter = nn.Parameter(torch.tensor([0.5, 0.5, 0.5]))
        self.img_mean.requires_grad = False
        self.img_std: nn.Parameter = nn.Parameter(torch.tensor([0.5, 0.5, 0.5]))
        self.img_std.requires_grad = False
        self.apply_image_norm: bool = apply_image_norm

        print(f"Model name: {self.model_name}, feature dimension: {self.feature_dim}")

        if self.frozen:
            for param in self.model.parameters():
                param.requires_grad = False

        assert feature_aggregation in [
            "map",
            "patches",
            "mean_2x2",
            "max_2x2",
            "mean_4x4",
            "max_4x4",
        ], f"feature_aggregation: {feature_aggregation} is not supported"
        self.feature_aggregation: str = feature_aggregation
        patch_num = (self.model.config.image_size // self.model.config.patch_size) ** 2
        if feature_aggregation == "mean_2x2":
            assert patch_num % 4 == 0, f"patch_num: {patch_num} is not divisible by 4"
            self.token_num: int = patch_num // 4
        elif feature_aggregation == "max_2x2":
            assert patch_num % 4 == 0, f"patch_num: {patch_num} is not divisible by 4"
            self.token_num = patch_num // 4
        elif feature_aggregation == "mean_4x4":
            assert patch_num % 16 == 0, f"patch_num: {patch_num} is not divisible by 16"
            self.token_num = patch_num // 16
        elif feature_aggregation == "max_4x4":
            assert patch_num % 16 == 0, f"patch_num: {patch_num} is not divisible by 16"
            self.token_num = patch_num // 16
        elif feature_aggregation == "patches":
            self.token_num = patch_num
        elif feature_aggregation == "map":
            self.token_num = 1
        else:
            raise ValueError(
                f"feature_aggregation: {feature_aggregation} is not supported"
            )

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"ViTImageEncoder trainable parameters: {trainable_params}, total parameters: {total_params}"
        )

    def get_siglip_output(self, x: torch.Tensor):
        """x.shape: (B, T, C, H, W), Output: pooler_output: (B, T, 1, D), patches_output: (B, T, P, D) where T is the number of frames and D is the feature dimension"""

        batch_size, obs_len, *image_shape = x.shape
        assert (
            obs_len == self.image_meta.length
        ), f"obs_len: {obs_len}, self.image_meta.length: {self.image_meta.length} are not consistent"
        assert (
            tuple(image_shape) == self.image_shape
        ), f"image_shape: {tuple(image_shape)}, self.image_shape: {self.image_shape}"

        x = x.reshape(batch_size * obs_len, *image_shape)
        if self.apply_image_norm:
            x = (x - self.img_mean[None, :, None, None]) / self.img_std[
                None, :, None, None
            ]

        raw_output = self.model(pixel_values=x)
        pooler_output = raw_output.pooler_output.reshape(batch_size, obs_len, 1, -1)
        patches_output = einops.rearrange(
            raw_output.last_hidden_state,
            "(b t) ... -> b t ...",
            b=batch_size,
            t=obs_len,
        )

        return pooler_output, patches_output

    def get_pooler_output(self, patches_output: torch.Tensor):
        """patches_output.shape: (B, T, P, D), Output: (B, T, 1, D)"""
        batch_size, obs_len, patch_num, feature_dim = patches_output.shape
        patches_output = einops.rearrange(patches_output, "b t p f -> (b t) p f")
        pooler_output: torch.Tensor = self.model.vision_model.head(patches_output)
        return pooler_output.reshape(batch_size, obs_len, 1, feature_dim)

    def mean_pool(self, x: torch.Tensor, ds: int):
        """
        x: (..., patch_num, feature_dim)
        ds: downsample factor
        return: (..., patch_num // ds**2, feature_dim)
        """
        patch_width = int(sqrt(x.shape[-2]))
        assert (
            x.shape[-2] == patch_width**2
        ), f"x.shape[-2]: {x.shape[-2]}, patch_width: {patch_width}"
        assert (
            patch_width % ds == 0
        ), f"patch_width: {patch_width} is not divisible by ds: {ds}"
        x = einops.rearrange(
            x,
            "... (p_h ds1 p_w ds2) f -> ... p_h ds1 p_w ds2 f",
            p_h=patch_width // ds,
            p_w=patch_width // ds,
            ds1=ds,
            ds2=ds,
        )
        x = torch.mean(
            x, dim=(-4, -2)
        )  # (..., patch_width//2, patch_width//2, feature_dim)
        x = einops.rearrange(
            x,
            "... p_h p_w f -> ... (p_h p_w) f",
            p_h=patch_width // ds,
            p_w=patch_width // ds,
        )
        return x

    def max_pool(self, x: torch.Tensor, ds: int):
        """
        x: (..., patch_num, feature_dim)
        ds: downsample factor
        return: (..., patch_num // ds**2, feature_dim)
        """
        patch_width = int(sqrt(x.shape[-2]))
        assert (
            x.shape[-2] == patch_width**2
        ), f"x.shape[-2]: {x.shape[-2]}, patch_width: {patch_width}"
        assert (
            patch_width % ds == 0
        ), f"patch_width: {patch_width} is not divisible by ds: {ds}"
        x = einops.rearrange(
            x,
            "... (p_h ds1 p_w ds2) f -> ... p_h p_w (ds1 ds2) f",
            p_h=patch_width // ds,
            p_w=patch_width // ds,
            ds1=ds,
            ds2=ds,
        )
        x = torch.max(
            x, dim=-2
        ).values  # (..., patch_width//2, patch_width//2, feature_dim)
        x = einops.rearrange(
            x,
            "... p_h p_w f -> ... (p_h p_w) f",
            p_h=patch_width // ds,
            p_w=patch_width // ds,
        )
        return x

    def forward(self, x: torch.Tensor):
        """
        x.shape: (batch_size, trajectory_len, C, H, W) or (batch_size, traj_num, traj_len, C, H, W)
        return: (batch_size, trajectory_len, token_num, feature_dim) or (batch_size, traj_num, traj_len, token_num, feature_dim)
        """

        batch_size = x.shape[0]

        assert len(x.shape) == 6 or len(x.shape) == 5, f"x.shape: {x.shape}"

        if len(x.shape) == 6:
            x = einops.rearrange(x, "b t l c h w -> (b t) l c h w")

        if self.feature_aggregation == "map":
            x = self.get_siglip_output(x)[0]
        elif self.feature_aggregation == "patches":
            x = self.get_siglip_output(x)[1]
        elif self.feature_aggregation == "mean_2x2":
            x = self.get_siglip_output(x)[
                1
            ]  # (batch_size*traj_num, traj_len, patch_num, feature_dim)
            x = self.mean_pool(x, 2)
        elif self.feature_aggregation == "max_2x2":
            x = self.get_siglip_output(x)[
                1
            ]  # (batch_size*traj_num, traj_len, patch_num, feature_dim)
            x = self.max_pool(x, 2)
        elif self.feature_aggregation == "mean_4x4":
            x = self.get_siglip_output(x)[
                1
            ]  # (batch_size*traj_num, traj_len, patch_num, feature_dim)
            x = self.mean_pool(x, 4)
        elif self.feature_aggregation == "max_4x4":
            x = self.get_siglip_output(x)[
                1
            ]  # (batch_size*traj_num, traj_len, patch_num, feature_dim)
            x = self.max_pool(x, 4)
        else:
            raise ValueError(
                f"feature_aggregation: {self.feature_aggregation} is not supported"
            )

        if batch_size != x.shape[0]:
            x = einops.rearrange(x, "(b t) l n f -> b t l n f", b=batch_size)

        return x


class SharedModelManager(nn.Module):
    """
    This class is used to manage the shared models for the image encoders. Use it as a singleton.
    """

    _instance = None
    _deepcopy_instance = None

    def __new__(cls, *args, **kwargs):
        # Make it a singleton
        if cls._instance is None:
            # object.__new__ accepts only the class; constructor arguments are
            # handled by __init__ after the singleton instance is returned.
            cls._instance = super().__new__(cls)
        return cls._instance

    def __deepcopy__(self, memo):
        """Will still create a new instance on the first deepcopy (Used for ema model)"""
        cls = self.__class__
        if cls._deepcopy_instance is None:
            result = super(cls, cls).__new__(cls)  # Will create a new instance
            memo[id(self)] = result

            for k, v in self.__dict__.items():
                setattr(result, k, copy.deepcopy(v, memo))
            cls._deepcopy_instance = result
        return cls._deepcopy_instance

    def __init__(self, cache_size: int | None = None):
        if not hasattr(self, "initialized"):
            super().__init__()
            self.vit_models: nn.ModuleDict = (
                nn.ModuleDict()
            )  # ViT backbone, usually shared
            self.vit_frozen: dict[str, bool] = {}
            # Legacy online checkpoints used a redundant frozen target DINO.
            # Aliases are metadata only, never duplicate registered modules.
            self.vit_aliases: dict[str, str] = {}
            self.map_models: nn.ModuleDict = (
                nn.ModuleDict()
            )  # Multihead Attention Pooling, usually different for each camera
            self.map_frozen: dict[str, bool] = {}
            # Include the model (and MAP head) identity in cache keys. This is
            # required when a process constructs policies with different
            # selectable visual backends one after another.
            self.patches_cache: dict[
                tuple[str, torch.Tensor], torch.Tensor
            ] = {}
            self.map_cache: dict[
                tuple[str, str, torch.Tensor], torch.Tensor
            ] = {}
            self.initialized: bool = True
            self.cache_size: int | None = None
        if cache_size is not None:
            assert (
                self.cache_size is None or self.cache_size == cache_size
            ), f"cache_size is already set to {self.cache_size}, but got {cache_size} for the same model"
            self.cache_size = cache_size

    def register_vit_model(
        self,
        model_name: str,
        frozen: bool,
        model: nn.Module | None = None,
        registry_name: str | None = None,
    ):
        """Register a ViT, optionally under a separate runtime key.

        The same pretrained checkpoint may be needed once as a trainable
        online encoder and once as a frozen future-target encoder.  Keeping a
        distinct registry key prevents the singleton manager from forcing the
        two copies to share the same ``requires_grad`` state.
        """
        key = registry_name or model_name
        if (
            key == "future_target_dinov2"
            and frozen
            and self.vit_frozen.get(model_name) is True
        ):
            self.vit_aliases[key] = model_name
            key = model_name

        if key in self.vit_models:
            assert (
                self.vit_frozen[key] == frozen
            ), f"Model {key} is already registered with frozen={self.vit_frozen[key]}, \
                    but got {frozen} for the same model"
            print(
                f"Model {key} is already registered with frozen={self.vit_frozen[key]}"
            )
        else:
            if not model:
                if "siglip" in model_name.lower():
                    from transformers import SiglipVisionModel

                    model: nn.Module = SiglipVisionModel.from_pretrained(model_name)
                elif "dinov2" in model_name.lower():
                    from transformers import AutoModel

                    model = AutoModel.from_pretrained(model_name)
                else:
                    raise ValueError(f"Model {model_name} is not supported")

            self.vit_models[key] = model
            self.vit_frozen[key] = frozen
            for param in self.vit_models[key].parameters():
                param.requires_grad = not frozen
            print(f"Model {model_name} registered as {key} with frozen={frozen}")

        if self._is_frozen_dino(key):
            self.vit_models[key].float().eval()
            if key == model_name:
                self.vit_aliases["future_target_dinov2"] = key
        return key

    def _is_frozen_dino(self, key: str) -> bool:
        config = getattr(self.vit_models[key], "config", None)
        return self.vit_frozen[key] and "dinov2" in str(
            getattr(config, "model_type", key)
        ).lower()

    def train(self, mode: bool = True):
        super().train(mode)
        for key in self.vit_models:
            if self._is_frozen_dino(key):
                self.vit_models[key].eval()
        return self

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Fold identical legacy target tensors into the shared backbone. A
        # divergent target cannot be discarded without changing the policy.
        for alias, canonical in self.vit_aliases.items():
            if alias in self.vit_models:
                continue
            old_prefix = prefix + "vit_models." + alias + "."
            new_prefix = prefix + "vit_models." + canonical + "."
            for key in list(state_dict):
                if not key.startswith(old_prefix):
                    continue
                new_key = new_prefix + key[len(old_prefix):]
                if new_key in state_dict and not torch.equal(state_dict[key], state_dict[new_key]):
                    error_msgs.append(
                        f"Cannot share legacy frozen DINO: {key} differs from {new_key}"
                    )
                else:
                    state_dict.setdefault(new_key, state_dict[key])
                del state_dict[key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def compile(self):
        for model in self.vit_models.values():
            model = cast(nn.Module, torch.compile(model))

    def register_map_model(
        self,
        model_name: str,
        key_name: str,
        frozen: bool,
        map_model: nn.Module | None = None,
    ):

        if key_name in self.map_models:
            assert (
                self.map_frozen[key_name] == frozen
            ), f"Model {model_name} key {key_name} is already registered with frozen={self.map_frozen[key_name]}, \
                    but got {frozen} for the same model"
            print(
                f"Model {model_name} key {key_name} is already registered with frozen={self.map_frozen[key_name]}"
            )
        else:
            if not map_model:
                if "siglip" in model_name:
                    from transformers import SiglipVisionModel

                    model: nn.Module = SiglipVisionModel.from_pretrained(model_name)
                    map_model = copy.deepcopy(model.vision_model.head)
                    del model
                else:
                    raise ValueError(f"Model {model_name} is not supported")

            self.map_models[key_name] = map_model
            self.map_frozen[key_name] = frozen

            for param in self.map_models[key_name].parameters():
                param.requires_grad = not frozen

            print(
                f"Model {model_name} key {key_name} is registered with frozen={frozen}"
            )

    def forward_patches(self, model_name: str, x: torch.Tensor):
        """
        x: (batch_size, C, H, W)
        return: (batch_size, patch_num, feature_dim)
        """
        assert self.cache_size is not None, "cache_size is not set"
        if model_name not in self.vit_models:
            model_name = self.vit_aliases.get(model_name, model_name)

        for (cached_model_name, key), value in self.patches_cache.items():
            if (
                cached_model_name == model_name
                and key.shape == x.shape
                and torch.allclose(key, x)
            ):
                # print(f"Patches cache hit for {model_name}, {x.shape=}")
                return value

        if len(self.patches_cache) >= self.cache_size and self.cache_size > 0:
            pop_key = next(iter(self.patches_cache.keys()))
            self.patches_cache.pop(pop_key)

        # print(f"Patches cache miss for {model_name}, {x.shape=}, {len(self.patches_cache)=}")
        if self._is_frozen_dino(model_name):
            # Freezing weights does not disable the trainer's BF16 autocast.
            # Match offline feature extraction in every frozen DINO caller.
            with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
                y = self.vit_models[model_name](x.float()).last_hidden_state
        else:
            y = self.vit_models[model_name](x).last_hidden_state
        self.patches_cache[(model_name, x)] = y
        return y

    def forward_map(self, model_name: str, key_name: str, x: torch.Tensor):
        """
        x: (batch_size, C, H, W)
        return: (batch_size, 1, feature_dim)
        """
        assert self.cache_size is not None, "cache_size is not set"
        batch_size = x.shape[0]
        for (cached_model_name, cached_key_name, key), value in (
            self.map_cache.items()
        ):
            if (
                cached_model_name == model_name
                and cached_key_name == key_name
                and key.shape == x.shape
                and torch.allclose(key, x)
            ):
                # print(f"MAP cache hit for {model_name}:{key_name}, {x.shape=}")
                return value

        if len(self.map_cache) >= self.cache_size and self.cache_size > 0:
            pop_key = next(iter(self.map_cache.keys()))
            self.map_cache.pop(pop_key)

        for (cached_model_name, key), value in self.patches_cache.items():
            if (
                cached_model_name == model_name
                and key.shape == x.shape
                and torch.allclose(key, x)
            ):
                patches_output = value
                # print(f"MAP cache miss for {model_name}:{key_name}, but patches cache hit, {x.shape=}, {len(self.map_cache)=}")
                break
        else:
            patches_output = self.forward_patches(model_name, x)
            # print(f"MAP cache miss for {model_name}:{key_name}, patches cache miss, {x.shape=}, {len(self.map_cache)=}")

        y: torch.Tensor = self.map_models[key_name](patches_output)
        feature_dim = y.shape[-1]
        y = y.reshape(batch_size, 1, feature_dim)
        self.map_cache[(model_name, key_name, x)] = y
        return y

    def clear_cache(self):
        self.patches_cache.clear()
        self.map_cache.clear()

    @classmethod
    def reset(cls):
        print(f"Resetting SharedModelManager")
        # Inside a instance, clear the map_models and vit_models
        if cls._instance is not None:
            cls._instance.map_models.clear()
            cls._instance.vit_models.clear()

        if cls._deepcopy_instance is not None:
            cls._deepcopy_instance.map_models.clear()
            cls._deepcopy_instance.vit_models.clear()

        cls._instance = None
        cls._deepcopy_instance = None


class SharedSiglipImageEncoder(BaseImageEncoder):
    def __init__(
        self,
        feature_aggregation: str,
        apply_image_norm: bool,
        individual_forward: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)


        self.individual_forward: bool = individual_forward
        """If true, will forward (then cache) images individually instead of the entire tensor. Used to speed up policy inference."""

        self.feature_aggregation: str = feature_aggregation

        self.manager = [
            SharedModelManager()
        ]  # Will bypass nn.Module.__setattr__ so it will not be saved multiple times
        self.manager[0].register_vit_model(self.model_name, self.frozen)

        if feature_aggregation == "map":
            self.manager[0].register_map_model(
                self.model_name, self.image_meta.name, self.frozen
            )

        config = self.manager[0].vit_models[self.model_name].config
        self.feature_dim: int = config.hidden_size
        patch_num = (config.image_size // config.patch_size) ** 2

        if feature_aggregation == "map":
            self.token_num: int = 1
        elif feature_aggregation == "patches":
            self.token_num = patch_num
        else:
            raise ValueError(
                f"feature_aggregation: {feature_aggregation} is not supported"
            )

        # For some unknown reason, this does not work with autocast
        # self.processor: SiglipImageProcessor = SiglipImageProcessor.from_pretrained(self.model_name)
        # Manually assign the mean and std for siglip
        self.img_mean: nn.Parameter = nn.Parameter(torch.tensor([0.5, 0.5, 0.5]))
        self.img_mean.requires_grad = False
        self.img_std: nn.Parameter = nn.Parameter(torch.tensor([0.5, 0.5, 0.5]))
        self.img_std.requires_grad = False
        self.apply_image_norm: bool = apply_image_norm

    def forward(self, x: torch.Tensor):
        """
        x.shape: (batch_size, traj_len, C, H, W) or (batch_size, traj_num, traj_len, C, H, W)
        """

        batch_size = x.shape[0]
        ndim = len(x.shape)
        if ndim == 6:
            traj_num = x.shape[1]
            traj_len = x.shape[2]
            x = einops.rearrange(x, "b t l c h w -> (b t l) c h w")

        elif ndim == 5:
            traj_num = 1
            traj_len = x.shape[1]
            x = einops.rearrange(x, "b t c h w -> (b t) c h w")

        else:
            raise ValueError(
                f"x.shape: {x.shape} should be (batch_size, traj_len, C, H, W) or (batch_size, traj_num, traj_len, C, H, W)"
            )
        
        if self.individual_forward:
            result_list: list[torch.Tensor] = []
            for i in range(traj_len):
                x_i = x[i:i+1, :, :, :]
                if self.feature_aggregation == "map":
                    x_i = self.manager[0].forward_map(self.model_name, self.image_meta.name, x_i)
                elif self.feature_aggregation == "patches":
                    x_i = self.manager[0].forward_patches(self.model_name, x_i)
                else:
                    raise ValueError(
                        f"feature_aggregation: {self.feature_aggregation} is not supported"
                    )
                result_list.append(x_i)
            x = torch.cat(result_list, dim=0)
        else:
            if self.feature_aggregation == "map":
                x = self.manager[0].forward_map(self.model_name, self.image_meta.name, x)
            elif self.feature_aggregation == "patches":
                x = self.manager[0].forward_patches(self.model_name, x)
            else:
                raise ValueError(
                    f"feature_aggregation: {self.feature_aggregation} is not supported"
                )

        if ndim == 6:
            x = einops.rearrange(
                x, "(b t l) n f -> b t l n f", b=batch_size, t=traj_num, l=traj_len
            )
        elif ndim == 5:
            x = einops.rearrange(x, "(b t) n f -> b t n f", b=batch_size, t=traj_len)

        return x


class SharedDinoV2ImageEncoder(BaseImageEncoder):
    """Shared frozen DINOv2 encoder with explicit patch/register handling.

    ``patches`` returns spatial patch tokens only. ``map`` and ``cls`` both
    return the single CLS token so existing policy YAML can use one common
    name for a backend-specific global image token (SigLIP MAP or DINO CLS).
    """

    def __init__(
        self,
        feature_aggregation: str,
        apply_image_norm: bool,
        resize_size: int | tuple[int, int] | list[int] = 224,
        individual_forward: bool = False,
        spatial_merge_size: int = 1,
        shared_model_key: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if feature_aggregation not in {"patches", "map", "cls"}:
            raise ValueError(
                "DINOv2 feature_aggregation must be 'patches', 'map', or "
                f"'cls', got {feature_aggregation!r}"
            )
        self.feature_aggregation = feature_aggregation
        self.apply_image_norm = apply_image_norm
        self.individual_forward = individual_forward
        if (
            isinstance(spatial_merge_size, bool)
            or not isinstance(spatial_merge_size, int)
            or spatial_merge_size <= 0
        ):
            raise ValueError("spatial_merge_size must be a positive integer")
        self.spatial_merge_size = spatial_merge_size
        self.shared_model_key = shared_model_key or self.model_name

        if isinstance(resize_size, int):
            resize_hw = (resize_size, resize_size)
        else:
            resize_hw = tuple(resize_size)
        if len(resize_hw) != 2 or any(size <= 0 for size in resize_hw):
            raise ValueError(
                "resize_size must be a positive integer or a two-element "
                f"sequence, got {resize_size!r}"
            )
        self.resize_size = cast(tuple[int, int], resize_hw)

        self.manager = [
            SharedModelManager()
        ]  # Bypass nn.Module registration; the manager owns the shared model.
        self.shared_model_key = self.manager[0].register_vit_model(
            self.model_name,
            self.frozen,
            registry_name=self.shared_model_key,
        )
        config = self.manager[0].vit_models[self.shared_model_key].config
        if "dinov2" not in str(getattr(config, "model_type", "")).lower():
            raise ValueError(
                f"Model {self.model_name!r} is not a DINOv2 backbone"
            )

        self.feature_dim = int(config.hidden_size)
        self.patch_size = int(config.patch_size)
        self.num_register_tokens = int(
            getattr(config, "num_register_tokens", 0)
        )
        height, width = self.resize_size
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                "DINOv2 resize dimensions must be divisible by patch_size "
                f"({self.resize_size} vs {self.patch_size})"
            )
        self.patch_token_num = (
            height // self.patch_size
        ) * (width // self.patch_size)
        patch_grid = int(sqrt(self.patch_token_num))
        if patch_grid * patch_grid != self.patch_token_num:
            raise ValueError(
                "DINOv2 spatial patch grid must be square for spatial merging, "
                f"got {self.patch_token_num} patches"
            )
        if patch_grid % self.spatial_merge_size != 0:
            raise ValueError(
                "spatial_merge_size must divide the DINOv2 patch grid, got "
                f"grid={patch_grid}, spatial_merge_size={self.spatial_merge_size}"
            )
        self.token_num = (
            self.patch_token_num // self.spatial_merge_size**2
            if feature_aggregation == "patches"
            else 1
        )

        # DINOv2 uses the standard ImageNet normalization. Buffers follow the
        # encoder across devices without being optimized or duplicated in the
        # shared backbone.
        self.register_buffer(
            "img_mean",
            torch.tensor([0.485, 0.456, 0.406]),
            persistent=True,
        )
        self.register_buffer(
            "img_std",
            torch.tensor([0.229, 0.224, 0.225]),
            persistent=True,
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-2:]) != self.resize_size:
            x = torch.nn.functional.interpolate(
                x,
                size=self.resize_size,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        if self.apply_image_norm:
            mean = self.img_mean.to(dtype=x.dtype)[None, :, None, None]
            std = self.img_std.to(dtype=x.dtype)[None, :, None, None]
            x = (x - mean) / std
        return x

    def _select_features(self, x: torch.Tensor) -> torch.Tensor:
        raw_tokens = self.manager[0].forward_patches(self.shared_model_key, x)
        expected_token_num = (
            1 + self.num_register_tokens + self.patch_token_num
        )
        if raw_tokens.shape[1] != expected_token_num:
            raise RuntimeError(
                "Unexpected DINOv2 token count: expected "
                f"{expected_token_num}, got {raw_tokens.shape[1]}"
            )
        if self.feature_aggregation == "patches":
            patches = raw_tokens[:, 1 + self.num_register_tokens :, :]
            return self.merge_spatial_patches(patches)
        return raw_tokens[:, :1, :]

    def merge_spatial_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Average each spatial ``size x size`` patch block.

        The cached RMBench features retain the original 256 spatial tokens, so
        this helper is also used by ``MultiTokenEncoder`` when it reads a
        cache.  The feature width stays unchanged while the token count is
        reduced by ``spatial_merge_size**2``.
        """

        if patches.shape[-2] != self.patch_token_num:
            raise ValueError(
                "DINOv2 patch tensor has an unexpected token count: expected "
                f"{self.patch_token_num}, got {patches.shape[-2]}"
            )
        size = self.spatial_merge_size
        if size == 1:
            return patches
        grid = int(sqrt(self.patch_token_num))
        patches = patches.reshape(
            *patches.shape[:-2],
            grid // size,
            size,
            grid // size,
            size,
            patches.shape[-1],
        )
        patches = patches.mean(dim=(-4, -2))
        return patches.reshape(
            *patches.shape[:-3],
            (grid // size) ** 2,
            patches.shape[-1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode 5-D online or 6-D multi-trajectory image batches."""

        batch_size = x.shape[0]
        ndim = x.ndim
        if ndim == 6:
            traj_num, traj_len = x.shape[1:3]
            expected_shape = self.image_shape
            if tuple(x.shape[3:]) != expected_shape:
                raise ValueError(
                    f"image shape {tuple(x.shape[3:])} does not match "
                    f"configured shape {expected_shape}"
                )
            x = einops.rearrange(x, "b t l c h w -> (b t l) c h w")
        elif ndim == 5:
            traj_num = 1
            traj_len = x.shape[1]
            expected_shape = self.image_shape
            if tuple(x.shape[2:]) != expected_shape:
                raise ValueError(
                    f"image shape {tuple(x.shape[2:])} does not match "
                    f"configured shape {expected_shape}"
                )
            x = einops.rearrange(x, "b l c h w -> (b l) c h w")
        else:
            raise ValueError(
                "DINOv2 input must have shape (B,L,C,H,W) or "
                f"(B,T,L,C,H,W), got {tuple(x.shape)}"
            )

        x = self._preprocess(x)
        if self.individual_forward:
            encoded = torch.cat(
                [self._select_features(x[index : index + 1]) for index in range(x.shape[0])],
                dim=0,
            )
        else:
            encoded = self._select_features(x)

        if ndim == 6:
            return einops.rearrange(
                encoded,
                "(b t l) n f -> b t l n f",
                b=batch_size,
                t=traj_num,
                l=traj_len,
            )
        return einops.rearrange(
            encoded,
            "(b l) n f -> b l n f",
            b=batch_size,
            l=traj_len,
        )
