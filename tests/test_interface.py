"""CPU-only checks; no generation claims are made by these tests."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("release_generate", ROOT / "generate.py")
generate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate)


class InterfaceTests(unittest.TestCase):
    def test_fixed_presets(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            for model, delta, steps in (("flux", 1.5, 28), ("sana", .5, 20), ("sd15", .75, 100)):
                args = generate.parser().parse_args(["--model", model, "--output", str(Path(directory) / model), "--dry-run"])
                config, prompts, output = generate.resolve(args)
                self.assertEqual(config["tau_relmax_offset"], delta)
                self.assertEqual(config["num_steps"], steps)
                self.assertEqual(config["k"], 4)
                self.assertTrue(config["reward_ramp"] and config["clone_dud"])
                self.assertFalse(output.exists())

    def test_delta_and_seed_only_override_requested_fields(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            args = generate.parser().parse_args(["--output", str(Path(directory) / "new"), "--delta", "1.25", "--seed", "10"])
            config, _, _ = generate.resolve(args)
            self.assertEqual(config["tau_relmax_offset"], 1.25)
            self.assertEqual(config["seed"], 10)
            self.assertEqual(config["score_steps"], "20,40,60,80,99")

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            args = generate.parser().parse_args(["--output", directory])
            with self.assertRaises(ValueError):
                generate.resolve(args)

    def test_invalid_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "input.jsonl"
            for pid in ("../escape", "x", "1234567890"):
                prompts.write_text(json.dumps({"prompt_id": pid, "prompt": "a test"}), encoding="utf-8")
                args = generate.parser().parse_args(["--prompts", str(prompts), "--output", str(Path(directory) / "new")])
                with self.assertRaises(ValueError):
                    generate.resolve(args)

    def test_no_arbitrary_method_flag(self):
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            generate.parser().parse_args(["--output", "new", "--lam_d", "0"])

    def test_worker_mapping(self):
        import sys
        with patch.dict(os.environ, {"SATISDIVE_REWARD_PYTHON": sys.executable, "CUDA_VISIBLE_DEVICES": "4,7", "HF_HOME": "test-cache"}, clear=True):
            config = {"worker_gpu": 1}
            generate.worker_environment("sana", config)
            self.assertEqual(config["worker_gpu"], "7")
            config = {"worker_gpu": 1}
            generate.worker_environment("flux", config)
            self.assertEqual(config["worker_gpu"], 1)

    def test_negative_nan_infinite_tolerance(self):
        import argparse
        for value in ("-1", "nan", "inf", "-inf"):
            with self.assertRaises(argparse.ArgumentTypeError):
                generate.positive(value)

    def test_no_missing_worker_or_outside_allocation(self):
        import sys
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            generate.worker_environment("sd15", {"worker_gpu": 0})
        with patch.dict(os.environ, {"SATISDIVE_REWARD_PYTHON": sys.executable, "CUDA_VISIBLE_DEVICES": "4", "HF_HOME": "test-cache"}, clear=True), self.assertRaises(ValueError):
            generate.worker_environment("flux", {"worker_gpu": 1})

    def test_worker_keeps_virtual_environment_executable_path(self):
        with patch.dict(os.environ, {"SATISDIVE_REWARD_PYTHON": sys.executable, "HF_HOME": "test-cache"}, clear=True), patch.object(Path, "resolve", side_effect=AssertionError("must not dereference venv interpreter")):
            generate.worker_environment("sd15", {"worker_gpu": 0})
            self.assertEqual(os.environ["SD15_IMAGEREWARD_PYTHON"], str(Path(sys.executable).absolute()))

    def test_prompt_record_must_be_object(self):
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "input.jsonl"
            prompts.write_text('["not", "a", "record"]', encoding="utf-8")
            args = generate.parser().parse_args(["--prompts", str(prompts), "--output", str(Path(directory) / "new")])
            with self.assertRaises(ValueError):
                generate.resolve(args)


if __name__ == "__main__":
    unittest.main()
