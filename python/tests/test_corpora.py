"""Check corpus boundaries against the tokenizer used by the studio and CLI."""

import unittest
from pathlib import Path

from seed_torch.model import encode_training

ROOT = Path(__file__).resolve().parents[2]


class CorpusTests(unittest.TestCase):
    def test_color_sessions_fit_context_and_terminate_every_answer(self):
        for name, count in [("color_reasoning_01", 12610), ("color_reasoning_02", 31173)]:
            text = (ROOT / "data" / "inputs" / f"{name}.txt").read_text()
            sessions = text.strip().split("\n\n\n")
            self.assertEqual(len(sessions), count)
            self.assertEqual(len(set(sessions)), count)
            # Check the entire file too: boundaries must survive concatenation.
            self.assertEqual(encode_training(text).count(257), text.count("Assistant:"))
            for session in sessions:
                tokens = encode_training(session)
                self.assertLessEqual(len(tokens), 256)
                self.assertEqual(tokens.count(257), session.count("Assistant:"))
                self.assertEqual(session.count("User:"), session.count("Assistant:"))


if __name__ == "__main__":
    unittest.main()
