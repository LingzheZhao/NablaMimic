"""The reference-cache manifest on a tiny synthetic cache (CPU, numpy only)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nablamimic import export_reference_manifest as erm
from nablamimic.motion import CACHE_KEYS, LAFAN1_DATASET, LAFAN1_REVISION, cache_file_name

BODIES = ("pelvis", "torso_link", "left_knee_link")


def write_clip(directory, name, frames, bodies=BODIES, **override):
    arrays = dict(qpos=np.zeros((frames, 9)), qvel=np.zeros((frames, 8)),
                  body_pos=np.zeros((frames, len(bodies), 3)), body_quat=np.ones((frames, len(bodies), 4)) * 0.5,
                  body_lin_vel=np.zeros((frames, len(bodies), 3)), body_ang_vel=np.zeros((frames, len(bodies), 3)))
    arrays.update(override)
    path = Path(directory) / cache_file_name(name, 50.0, bodies)
    np.savez(path, **arrays)
    return path


def write_checksums(directory, names):
    path = Path(directory) / "checksums.sha256"
    path.write_text("# comment line\n" + "".join(f"{'0' * 64}  {n}.csv\n" for n in names))
    return path


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache = self.root / "reference-cache"
        self.cache.mkdir()
        # created out of name order: the manifest order must not follow file creation
        for name, frames in (("walk1_subject1", 5), ("dance1_subject1", 3), ("fight1_subject2", 4)):
            write_clip(self.cache, name, frames)
        self.checksums = write_checksums(self.root, ["walk1_subject1", "fight1_subject2", "dance1_subject1"])
        self.out = self.root / "manifest.json"

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, clips=("all",), **kw):
        return erm.build_manifest(self.cache, list(clips), "reference-test", self.out, BODIES,
                                  checksums=self.checksums, **kw)

    def test_all_clips_in_sorted_order_with_rows(self):
        m = self.build()
        self.assertEqual(m["format"], erm.FORMAT)
        self.assertEqual([c["name"] for c in m["clips"]], ["dance1_subject1", "fight1_subject2", "walk1_subject1"])
        self.assertEqual([c["frames"] for c in m["clips"]], [3, 4, 5])
        self.assertEqual([(c["first_row"], c["last_row"]) for c in m["clips"]], [(0, 2), (3, 6), (7, 11)])
        self.assertEqual(m["total_frames"], 12)
        self.assertEqual([c["cache_file"] for c in m["clips"]],
                         [cache_file_name(n, 50.0, BODIES) for n in ("dance1_subject1", "fight1_subject2",
                                                                     "walk1_subject1")])
        self.assertEqual((m["clip_selection"], m["clip_order"]), ("all", "sorted_clip_names"))
        self.assertEqual(m["cache_directory"], "reference-cache")  # beside the manifest: relative
        self.assertEqual(m["bodies"], list(BODIES))
        self.assertEqual(m["source_dataset"]["repository"], LAFAN1_DATASET)
        self.assertEqual(m["source_dataset"]["revision"], LAFAN1_REVISION)
        self.assertEqual(m["clips"][0]["source_file"], "g1/dance1_subject1.csv")
        self.assertEqual(m["arrays"]["qpos"]["shape_after_frames"], [9])
        self.assertEqual(m["arrays"]["body_quat"]["shape_after_frames"], [3, 4])
        self.assertEqual(set(m["arrays"]), set(CACHE_KEYS))

    def test_listed_clips_keep_their_order(self):
        m = self.build(["walk1_subject1", "dance1_subject1"])
        self.assertEqual([c["name"] for c in m["clips"]], ["walk1_subject1", "dance1_subject1"])
        self.assertEqual((m["clip_selection"], m["clip_order"]), ("listed", "as_listed"))
        self.assertEqual(m["clips"][1]["first_row"], 5)

    def test_cache_elsewhere_is_absolute(self):
        out = self.root / "elsewhere" / "manifest.json"
        out.parent.mkdir()
        m = erm.build_manifest(self.cache, ["all"], "reference-test", out, BODIES, checksums=self.checksums)
        self.assertEqual(m["cache_directory"], str(self.cache.resolve()))
        self.assertEqual(Path(out.parent, m["cache_directory"]), self.cache.resolve())

    def test_refusals(self):
        with self.assertRaises(ValueError):
            self.build(["run1_subject2"])  # no recorded dataset file
        with self.assertRaises(ValueError):
            self.build(["all", "walk1_subject1"])
        with self.assertRaises(ValueError):
            erm.build_manifest(self.cache, ["all"], "", self.out, BODIES, checksums=self.checksums)
        write_checksums(self.root, ["walk1_subject1", "fight1_subject2", "dance1_subject1", "run1_subject2"])
        with self.assertRaises(FileNotFoundError):
            self.build()

    def test_clip_content_refusals(self):
        cases = [dict(extra=np.zeros(3)), dict(body_pos=np.zeros((4, 2, 3))), dict(qvel=np.zeros((4, 7))),
                 dict(body_lin_vel=np.full((4, 3, 3), np.nan)), dict(body_quat=np.ones((4, 3, 3)))]
        for override in cases:
            with self.subTest(keys=sorted(override)):
                write_clip(self.cache, "fight1_subject2", 4, **override)
                with self.assertRaises(ValueError):
                    self.build()

    def test_written_once(self):
        erm.write_json_once(self.build(), self.out)
        self.assertEqual(erm.read_manifest(self.out)["clips"][0]["name"], "dance1_subject1")
        with self.assertRaises(FileExistsError):
            erm.write_json_once(self.build(), self.out)
        self.out.write_text(json.dumps({"format": "other"}))
        with self.assertRaises(ValueError):
            erm.read_manifest(self.out)


class RecordedIdentityTests(unittest.TestCase):
    def test_download_script_fetches_the_declared_revision(self):
        erm.check_download_identity()

    def test_recorded_clips(self):
        names = erm.recorded_clip_names()
        self.assertEqual(len(names), 40)
        self.assertEqual(names, sorted(names))


if __name__ == "__main__":
    unittest.main()
