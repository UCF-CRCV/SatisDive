"""Verify the HPSv3 worker bypasses the inference-only reward() API."""
import importlib.util
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU PyTorch is sufficient for this contract test")
class RewardContractTests(unittest.TestCase):
    def test_image_prompt_order_and_gradient(self):
        path = Path(__file__).resolve().parents[1] / "implementation/flux/rewards/workers/hpsv3_grad_worker.py"
        spec = importlib.util.spec_from_file_location("hpsv3_contract_worker", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class FakeInferencer:
            def prepare_batch(self, images, prompts):
                self.received = images, prompts
                return {"pixels": images[0]}

            def model(self, *, return_dict, pixels):
                return {"logits": pixels.mean().reshape(1, 1)}

            def reward(self, *args):
                raise AssertionError("must not call the inference-only wrapper")

        inferencer = FakeInferencer()
        image = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2).requires_grad_(True)
        reward = module._differentiable_reward(inferencer, image, "test prompt")
        gradient = torch.autograd.grad(reward[0, 0], image)[0]
        self.assertIs(inferencer.received[0][0], image)
        self.assertEqual(inferencer.received[1], ["test prompt"])
        torch.testing.assert_close(gradient, torch.full_like(image, 1 / image.numel()))


if __name__ == "__main__":
    unittest.main()
