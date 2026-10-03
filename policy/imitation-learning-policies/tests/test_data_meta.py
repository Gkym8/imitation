import unittest

from imitation_learning.common.dataclasses import DataMeta


class DataMetaTest(unittest.TestCase):
    def test_numeric_string_shape_and_resize_are_normalized(self) -> None:
        meta = DataMeta(
            name="head_camera",
            shape=[3, "224", "224"],
            data_type="image",
            length="1",
            normalizer="identity",
            augmentation=[
                {
                    "name": "Resize",
                    "size": ["224", "224"],
                    "antialias": True,
                }
            ],
            source_entry_names=["head_camera"],
        )

        self.assertEqual(meta.shape, (3, 224, 224))
        self.assertEqual(meta.length, 1)
        self.assertEqual(meta.augmentation[0]["size"], [224, 224])


if __name__ == "__main__":
    unittest.main()
