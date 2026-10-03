import inspect
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from rmbench_model import RMBenchMemoryPolicy


class RMBenchCheckpointConfigTest(unittest.TestCase):
    def test_adapter_has_no_runtime_memory_override_arguments(self) -> None:
        parameters = inspect.signature(RMBenchMemoryPolicy).parameters

        self.assertNotIn("inference_config", parameters)
        self.assertNotIn("recent_slot_protection_num", parameters)

    def test_deploy_config_has_no_inference_override_block(self) -> None:
        policy_root = Path(__file__).resolve().parents[1]
        config = OmegaConf.load(policy_root / "deploy_policy.yml")

        self.assertNotIn("inference", config)
        self.assertNotIn("recent_slot_protection_num", config)


if __name__ == "__main__":
    unittest.main()
