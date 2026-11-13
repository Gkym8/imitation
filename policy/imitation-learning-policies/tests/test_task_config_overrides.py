"""Exercise task YAML inheritance without constructing a training workspace."""
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import hydra
from hydra.errors import MissingConfigException
from omegaconf import OmegaConf, open_dict

from imitation_learning.utils.config_utils import compose_hydra_config, compose_training_config
from robot_utils.config_utils import register_resolvers


ROOT = Path(__file__).resolve().parents[1]
POLICY = 'diffusion_memory_transformer'


class TaskConfigOverridesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not OmegaConf.has_resolver('eval'):
            register_resolvers()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config_dir = Path(self.temp.name) / 'configs'
        shutil.copytree(ROOT / 'imitation_learning/configs', self.config_dir)
        (self.config_dir / 'task/rearrange_blocks.yaml').unlink(missing_ok=True)
        self.hydra = hydra.initialize_config_dir(config_dir=str(self.config_dir), version_base=None)
        self.hydra.__enter__()
        self.addCleanup(self.hydra.__exit__, None, None, None)

    def task(self, text, name='rearrange_blocks'):
        (self.config_dir / 'task' / f'{name}.yaml').write_text(text)

    def launch(self, *overrides):
        return hydra.compose('train_policy', overrides=[
            '+task_name=rearrange_blocks', '+policy_name=' + POLICY,
            '+task_yaml_fallback=rmbench', '+project_name=rmbench',
            '+logger_project_name=rmbench', '+run_name=config-test',
            '+run_name_suffix=_config-test', '+train_server_name=localhost',
            '+date_str=2026-11-09', '+time_str=00-00-00',
            '+workspace.train_dataset.name=rearrange_blocks-demo_clean-50',
            '+workspace.train_dataset.root_dir=/tmp/test-policy-data',
            '+workspace.train_dataset.compressed_dir=/tmp/test-policy-data',
            '+workspace.train_dataset.normalizer_dir=/tmp/test-policy-data',
            '+workspace.train_dataset.source_data_meta.agent_pos.shape=[14]',
            '+workspace.train_dataset.source_data_meta.action.shape=[14]',
            '+workspace.train_dataset.output_data_meta.agent_pos.shape=[14]',
            '+workspace.train_dataset.output_data_meta.action.shape=[14]',
            *overrides,
        ])

    def test_missing_task_file_matches_existing_rmbench_training_config(self):
        launch = self.launch()
        actual = compose_training_config(launch)
        # The original launcher did not have this new metadata argument.
        with open_dict(launch):
            del launch['task_yaml_fallback']
        expected = OmegaConf.merge(compose_hydra_config('rmbench', POLICY), launch)
        self.assertEqual(actual.task_yaml_name, 'rmbench')
        self.assertEqual(actual.task_name, 'rearrange_blocks')
        self.assertEqual(OmegaConf.to_container(actual.workspace, resolve=True),
                         OmegaConf.to_container(expected.workspace, resolve=True))
        self.assertEqual(OmegaConf.to_container(actual.visual_encoder, resolve=True),
                         OmegaConf.to_container(expected.visual_encoder, resolve=True))

    def test_partial_task_file_overrides_only_supplied_keys(self):
        self.task('''visual_encoder:
  spatial_merge_size: [4, 8]
workspace:
  model:
    denoising_network_partial:
      shadow_gate_scale: 0.1
  trainer:
    num_epochs: 401
    optimizer_partial:
      lr: 0.0002
''')
        cfg = compose_training_config(self.launch())
        baseline = compose_hydra_config('rmbench', POLICY)
        self.assertEqual(cfg.task_yaml_name, 'rearrange_blocks')
        self.assertEqual(cfg.dataset_type, 'rmbench')
        self.assertEqual(list(cfg.visual_encoder.spatial_merge_size), [4, 8])
        self.assertEqual(cfg.visual_encoder.model_name, baseline.visual_encoder.model_name)
        self.assertEqual(cfg.visual_encoder.learned_query_merge, baseline.visual_encoder.learned_query_merge)
        self.assertEqual(cfg.workspace.trainer.num_epochs, 401)
        self.assertEqual(cfg.workspace.trainer.optimizer_partial.lr, 0.0002)
        self.assertEqual(cfg.workspace.trainer.optimizer_partial.betas,
                         baseline.workspace.trainer.optimizer_partial.betas)
        self.assertEqual(cfg.workspace.model.denoising_network_partial.shadow_gate_scale, 0.1)
        self.assertEqual(cfg.workspace.model.denoising_network_partial.max_history_len, 8)
        self.assertEqual(cfg.workspace.train_dataset.name, 'rearrange_blocks-demo_clean-50')

    def test_false_zero_and_list_values_replace_defaults(self):
        self.task('''visual_encoder:
  learned_query_merge: false
workspace:
  train_dataset:
    image_keys: [left_camera, right_camera]
  model:
    denoising_network_partial:
      shadow_horizon: 0
''')
        cfg = compose_training_config(self.launch())
        self.assertIs(cfg.visual_encoder.learned_query_merge, False)
        self.assertEqual(cfg.workspace.model.denoising_network_partial.shadow_horizon, 0)
        self.assertEqual(list(cfg.workspace.train_dataset.image_keys), ['left_camera', 'right_camera'])
        self.assertEqual(cfg.workspace.model.global_cond_encoder.image_encoder_partial.spatial_merge_size,
                         cfg.visual_encoder.spatial_merge_size)

    def test_command_line_overrides_remain_highest_priority(self):
        self.task('workspace:\n  trainer:\n    num_epochs: 401\n    optimizer_partial:\n      lr: 0.0002\n')
        cfg = compose_training_config(self.launch(
            '+workspace.trainer.num_epochs=501', '+workspace.trainer.optimizer_partial.lr=0.0007'))
        self.assertEqual(cfg.workspace.trainer.num_epochs, 501)
        self.assertEqual(cfg.workspace.trainer.optimizer_partial.lr, 0.0007)

    def test_empty_task_mapping_inherits_all_rmbench_settings(self):
        self.task('{}\n')
        cfg = compose_training_config(self.launch())
        baseline = compose_hydra_config('rmbench', POLICY)
        self.assertEqual(cfg.task_yaml_name, 'rearrange_blocks')
        self.assertEqual(OmegaConf.to_container(cfg.visual_encoder, resolve=False),
                         OmegaConf.to_container(baseline.visual_encoder, resolve=False))
        self.assertEqual(cfg.workspace.trainer.num_epochs, baseline.workspace.trainer.num_epochs)

    def test_saved_config_contains_effective_values_and_actual_task_yaml_name(self):
        self.task('visual_encoder:\n  spatial_merge_size: [8, 4]\n')
        cfg = compose_training_config(self.launch())
        saved = OmegaConf.to_yaml(cfg, resolve=False)
        restored = OmegaConf.create(saved)
        cfg.workspace.cfg_str_unresolved = saved
        OmegaConf.resolve(cfg)
        self.assertEqual(restored.task_yaml_name, 'rearrange_blocks')
        self.assertEqual(restored.task_name, 'rearrange_blocks')
        self.assertEqual(list(restored.visual_encoder.spatial_merge_size), [8, 4])
        self.assertEqual(restored.workspace.model.denoising_network_partial.spatial_merge_size, [8, 4])
        self.assertEqual(restored.workspace.model.global_cond_encoder.image_encoder_partial.spatial_merge_size,
                         1 if restored.visual_encoder.learned_query_merge else [8, 4])
        self.assertEqual(restored.workspace.model.future_img_feature_encoder.image_encoder_partial.spatial_merge_size, [8, 4])

    def test_training_entry_point_saves_and_passes_merged_config(self):
        self.task('workspace:\n  trainer:\n    num_epochs: 401\n')
        output = Path(self.temp.name) / 'training-output'
        launch = self.launch('+workspace.trainer.output_dir=' + str(output))
        # Run the real config/save entry point, with workspace construction and
        # training replaced so no dataset, GPU or backbone is loaded.
        with mock.patch('robot_utils.config_utils.register_resolvers'):
            entry = runpy.run_path(str(ROOT / 'scripts/train_policy.py'))
        workspace = mock.Mock()
        with mock.patch.object(entry['mp'], 'set_start_method'), \
             mock.patch('hydra.utils.instantiate', return_value=workspace) as instantiate:
            entry['main'].__wrapped__(launch)
        workspace.train.assert_called_once_with()
        self.assertEqual(instantiate.call_args.args[0].trainer.num_epochs, 401)
        saved = OmegaConf.load(output / ('rearrange_blocks_' + POLICY + '.yaml'))
        self.assertEqual(saved.task_yaml_name, 'rearrange_blocks')
        self.assertEqual(saved.task_name, 'rearrange_blocks')
        self.assertEqual(saved.workspace.trainer.num_epochs, 401)
        checkpoint_cfg = OmegaConf.create(saved.workspace.cfg_str_unresolved)
        self.assertEqual(checkpoint_cfg.task_yaml_name, 'rearrange_blocks')
        self.assertEqual(checkpoint_cfg.workspace.trainer.num_epochs, 401)

    def test_existing_file_with_broken_defaults_does_not_silently_fall_back(self):
        self.task('defaults:\n  - missing_required_task_config\n  - _self_\n')
        with self.assertRaises(MissingConfigException) as caught:
            compose_training_config(self.launch())
        self.assertIn('missing_required_task_config', caught.exception.missing_cfg_file)

    def test_existing_file_with_bad_yaml_does_not_silently_fall_back(self):
        self.task('visual_encoder: [invalid\n')
        with self.assertRaises(Exception):
            compose_training_config(self.launch())

    def test_task_file_can_also_use_explicit_hydra_inheritance(self):
        self.task('defaults:\n  - rmbench\n  - _self_\nworkspace:\n  trainer:\n    num_epochs: 401\n')
        cfg = compose_training_config(self.launch())
        self.assertEqual(cfg.dataset_type, 'rmbench')
        self.assertEqual(cfg.workspace.trainer.num_epochs, 401)
        baseline = compose_hydra_config('rmbench', POLICY)
        self.assertEqual(cfg.visual_encoder.frozen, baseline.visual_encoder.frozen)

    def test_explicit_task_yaml_selection_is_preserved(self):
        self.task('workspace:\n  trainer:\n    num_epochs: 401\n')
        cfg = compose_training_config(self.launch('+task_yaml_name=rmbench'))
        self.assertEqual(cfg.task_yaml_name, 'rmbench')
        self.assertEqual(cfg.workspace.trainer.num_epochs, 301)

    def test_non_rmbench_tasks_keep_original_config_selection(self):
        baseline = compose_hydra_config('mikasa_remember_color_3', POLICY)
        cfg = compose_training_config(OmegaConf.create({
            'task_name': 'mikasa_remember_color_3', 'policy_name': POLICY,
        }))
        self.assertEqual(cfg.dataset_type, 'mikasa')
        self.assertEqual(cfg.task_yaml_name, 'mikasa_remember_color_3')
        self.assertEqual(OmegaConf.to_container(cfg.workspace, resolve=False),
                         OmegaConf.to_container(baseline.workspace, resolve=False))
        self.assertEqual(cfg.visual_encoder.model_name, baseline.visual_encoder.model_name)
        self.assertNotIn('state_query_merge', cfg.visual_encoder)

    def test_missing_file_without_fallback_keeps_original_error(self):
        with self.assertRaises(MissingConfigException):
            compose_hydra_config('missing_task_config', POLICY)


class TrainingLauncherConfigTest(unittest.TestCase):
    def test_single_and_multiple_gpu_commands_enable_task_inheritance(self):
        for gpu_ids in ('0', '0,1'):
            with self.subTest(gpu_ids=gpu_ids), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                launcher = root / 'policy'
                launcher.mkdir()
                shutil.copy2(ROOT / 'train.sh', launcher / 'train.sh')
                data = root / 'data'
                data.mkdir()
                (data / 'rearrange_blocks-demo_clean-50.zarr').mkdir()
                (data / 'rearrange_blocks-demo_clean-50_normalizer.json').write_text('{}')
                captured = root / 'argv.json'
                fake_python = root / 'python'
                fake_python.write_text('''#!/usr/bin/env python3
import json, os, sys
if sys.argv[1:2] != ['-c']:
    with open(os.environ['CAPTURED_ARGV'], 'w') as f:
        json.dump(sys.argv[1:], f)
''')
                fake_python.chmod(0o755)
                env = dict(os.environ, POLICY_PYTHON=str(fake_python),
                           RMBENCH_PYTHON=str(fake_python), RMBENCH_PROCESSED_DATA_DIR=str(data),
                           CAPTURED_ARGV=str(captured))
                result = subprocess.run(['bash', str(launcher / 'train.sh'), 'rearrange_blocks',
                                         'demo_clean', '50', '0', '14', gpu_ids],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = json.loads(captured.read_text())
                self.assertIn('+task_name=rearrange_blocks', args)
                self.assertIn('+task_yaml_fallback=rmbench', args)
                self.assertFalse(any(arg.startswith('+task_yaml_name=') for arg in args))
                self.assertIn('+workspace.train_dataset.name=rearrange_blocks-demo_clean-50', args)


if __name__ == '__main__':
    unittest.main()
