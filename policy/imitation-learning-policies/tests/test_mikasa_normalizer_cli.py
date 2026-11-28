from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import json
import unittest

import zarr

import fit_mikasa_normalizer as cli
import test_mikasa_training as fixture


class MikasaNormalizerCLITest(unittest.TestCase):
    @contextmanager
    def dataset(self):
        with fixture.MikasaTrainingTest().dataset_config() as (cfg, location):
            # Two episodes keep the real default 0.98 training split nonempty.
            store = zarr.open_group(str(location), mode="a")
            zarr.copy(store["episode_0"], store, name="episode_1")
            store.attrs["episode_frame_nums"] = {"0": 43, "1": 43}
            yield cfg, location

    def test_task_directory_and_zarr_path_work_before_feature_extraction(self):
        for direct_zarr in (False, True):
            with self.subTest(direct_zarr=direct_zarr), self.dataset() as (_, location):
                store = zarr.open_group(str(location), mode="a")
                del store.attrs["dinov2_patch_feature_cache"]
                del store["episode_0/third_person_camera_patch_feature"]
                argument = location if direct_zarr else location.parent
                with redirect_stdout(io.StringIO()):
                    result = cli.main([str(argument), "mikasa_remember_color_3"])
                self.assertEqual(result, location.parents[1] / "RememberColor3-v0_normalizer.json")
                self.assertEqual(set(json.loads(result.read_text())), {"robot0_8d", "action0_8d"})

    def test_cli_preserves_old_absolute_action_statistics(self):
        with self.dataset() as (_, location):
            path = location.parents[1] / "RememberColor3-v0_normalizer.json"
            original = {"scale": [7.0] * 8, "offset": [-1.0] * 8}
            path.write_text(json.dumps({"action0_8d": original}))
            with redirect_stdout(io.StringIO()):
                cli.main([str(location.parent), "mikasa_remember_color_3"])
            result = json.loads(path.read_text())
            self.assertEqual(result["action0_8d"], original)
            self.assertIn("robot0_8d", result)
            self.assertEqual(len(list(path.parent.glob(path.name + ".backup.*"))), 1)

    def test_mismatched_task_is_rejected_before_writing_statistics(self):
        with self.dataset() as (_, location):
            message = io.StringIO()
            with redirect_stderr(message), self.assertRaises(SystemExit) as error:
                cli.main([str(location.parent), "mikasa_remember_color_9"])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("does not match task", message.getvalue())
            self.assertEqual(list(location.parents[1].glob("*_normalizer.json")), [])


if __name__ == "__main__":
    unittest.main()
