import io
import json
import unittest

from seed_torch.progress import Progress, duration


class Terminal(io.StringIO):
    def isatty(self):
        return True


class ProgressTests(unittest.TestCase):
    def sample(self, **overrides):
        return {
            "step": 100,
            "completed": 100,
            "total": 6000,
            "loss": 1.23456,
            "validationLoss": 1.45678,
            "estimatedRemainingSeconds": 125,
            **overrides,
        }

    def test_terminal_replaces_line_and_finishes_once(self):
        stream = Terminal()
        progress = Progress(stream)
        progress.report(self.sample(estimatedRemainingSeconds=None))
        progress.report(self.sample())
        self.assertNotIn("\n", stream.getvalue())
        self.assertEqual(stream.getvalue().count("\r\033[2K"), 2)
        self.assertIn("remaining calculating", stream.getvalue())
        self.assertIn(
            "100/6000 · loss 1.2346 · validation 1.4568 · remaining 2m 5s", stream.getvalue()
        )
        progress.finish()
        progress.finish()
        self.assertTrue(stream.getvalue().endswith("\n"))
        self.assertEqual(stream.getvalue().count("\n"), 1)

    def test_redirected_output_keeps_json_lines_without_escape_codes(self):
        stream = io.StringIO()
        progress = Progress(stream)
        progress.report(self.sample())
        progress.report(self.sample(completed=200))
        progress.finish()
        self.assertNotIn("\r", stream.getvalue())
        self.assertNotIn("\033", stream.getvalue())
        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual([record["completed"] for record in records], [100, 200])

    def test_duration(self):
        self.assertEqual(duration(0), "0s")
        self.assertEqual(duration(60), "1m 0s")
        self.assertEqual(duration(3661), "1h 1m")
