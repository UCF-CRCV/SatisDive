"""Test the no-weight reward preflight protocol without external dependencies."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_environment import PREFIX, check_worker


class EnvironmentTests(unittest.TestCase):
    def test_worker_success_and_offline_isolation(self):
        result = subprocess.CompletedProcess([], 0, "import message\n" + PREFIX + json.dumps({"python": "3.12"}), "")
        with patch("check_environment.subprocess.run", return_value=result) as run:
            self.assertEqual(check_worker("venv/bin/python", "flux"), {"python": "3.12"})
        command = run.call_args.args[0]
        self.assertEqual(command[0], "venv/bin/python")
        self.assertEqual(command[-2:], ["--worker", "flux"])
        env = run.call_args.kwargs["env"]
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")

    def test_worker_failure(self):
        result = subprocess.CompletedProcess([], 1, "", "missing package")
        with patch("check_environment.subprocess.run", return_value=result), self.assertRaisesRegex(ValueError, "missing package"):
            check_worker("python", "sd15")

    def test_worker_timeout(self):
        with patch("check_environment.subprocess.run", side_effect=subprocess.TimeoutExpired([], 90)), self.assertRaisesRegex(ValueError, "no generation started"):
            check_worker("python", "sana")

    def test_worker_ambiguous_output(self):
        result = subprocess.CompletedProcess([], 0, PREFIX + "{}\n" + PREFIX + "{}", "")
        with patch("check_environment.subprocess.run", return_value=result), self.assertRaisesRegex(ValueError, "unambiguous"):
            check_worker("python", "flux")


if __name__ == "__main__":
    unittest.main()
