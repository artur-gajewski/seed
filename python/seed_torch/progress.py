"""Terminal progress that leaves redirected JSON logs machine-readable."""

import json
import sys


def duration(seconds):
    seconds = max(0, int(seconds + 0.5))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


class Progress:
    def __init__(self, stream=None):
        self.stream = sys.stdout if stream is None else stream
        self.active = False

    def report(self, progress):
        if self.stream.isatty():
            remaining = progress["estimatedRemainingSeconds"]
            eta = "calculating" if remaining is None else duration(remaining)
            validation = progress["validationLoss"]
            validation_text = "pending" if validation is None else f"{validation:.4f}"
            line = (
                f"{progress['completed']}/{progress['total']}"
                f" · loss {progress['loss']:.4f}"
                f" · validation {validation_text} · remaining {eta}"
            )
            self.stream.write("\r\033[2K" + line)
            self.active = True
        else:
            self.stream.write(json.dumps(progress) + "\n")
        self.stream.flush()

    def finish(self):
        if self.active:
            self.stream.write("\n")
            self.stream.flush()
            self.active = False
