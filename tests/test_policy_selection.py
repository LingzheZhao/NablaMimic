import unittest
from nablamimic.params import evaluation_actor


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.live = {"enc": 1, "dec": 2}
        self.ema = {"enc": 3, "dec": 4}
        self.ck = dict(actor=self.live, actor_ema=self.ema, ema_updates=4,
                       completed_updates=4, evaluation_policy="ema")

    def test_selection_uses_whole_actor(self):
        self.assertIs(evaluation_actor(self.ck, "ema"), self.ema)
        self.assertIs(evaluation_actor(self.ck, "live"), self.live)

    def test_no_silent_live_fallback(self):
        del self.ck["actor_ema"]
        with self.assertRaises(KeyError):
            evaluation_actor(self.ck, "ema")

    def test_mismatch_rejected(self):
        self.ck["ema_updates"] = 3
        with self.assertRaises(ValueError):
            evaluation_actor(self.ck, "ema")

    def test_export_label_is_not_update_count(self):
        self.ck["iter"] = 3
        self.assertIs(evaluation_actor(self.ck, "ema"), self.ema)

    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            evaluation_actor(self.ck, "auto")

if __name__ == "__main__":
    unittest.main()
