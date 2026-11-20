import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

PLUGIN = Path(__file__).parent / "hermes-plugins" / "spark-media"
spec = importlib.util.spec_from_file_location("spark_media_test", PLUGIN / "__init__.py",
                                             submodule_search_locations=[str(PLUGIN)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)


class MediaWorkerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.inbox = plugin.Inbox(self.root)

    def test_last_photo_cannot_be_replaced_by_replayed_older_message(self):
        first = self.cache / "img_0123456789ab.jpg"
        second = self.cache / "img_abcdef012345.png"
        for picture in (first, second):
            picture.write_bytes(b"image")
        self.inbox.remember(51, second, self.cache)
        self.inbox.remember(50, first, self.cache)
        self.assertEqual(self.inbox.latest(self.cache), second)

    def test_forged_outside_path_and_symlink_are_refused(self):
        outside = self.root / "img_abcdef012345.jpg"
        outside.write_bytes(b"image")
        symlink = self.cache / outside.name
        symlink.symlink_to(outside)
        for candidate in (outside, symlink):
            self.inbox.remember(10, candidate, self.cache)
            self.assertIsNone(self.inbox.latest(self.cache))

    def test_video_prompt_is_data_and_does_not_change_export_path(self):
        graph = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}},
            "2": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "old"}},
        }
        (self.root / "ltx23-smoke-prompt.json").write_text(json.dumps(graph))
        message = "Test literal characters ../ `echo foo` and $(uname)"
        result = plugin.MediaWorkers(self.root).load_workflow("video", message)
        self.assertEqual(result["1"]["inputs"]["text"], message)
        self.assertEqual(result["2"]["inputs"]["filename_prefix"], "video")

    def test_clear_media_requests_choose_the_right_worker(self):
        for message, name in (
            ("Turn this photo into a textured 3D GLB and send me the file.", "create_textured_glb"),
            ("Create an untextured mesh from the last photo.", "create_geometry_glb"),
            ("Generate a five-second video of a rolling ball.", "generate_spark_video"),
            ("請把這張相片轉成有貼圖的3D模型", "create_textured_glb"),
        ):
            with self.subTest(message=message):
                self.assertEqual(plugin.explicit_media_intent(message)[0], name)
        for message in ("Read the printed label.", "Please don't generate a video.",
                        "Can you explain how to create a GLB?", "What video tools are available?"):
            self.assertIsNone(plugin.explicit_media_intent(message))


if __name__ == "__main__":
    unittest.main()
