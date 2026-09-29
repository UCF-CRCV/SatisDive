# SatisDive

Minimal generation code for **Traversing the Satisfaction--Diversity Frontier in Text-to-Image Diffusion**. SatisDive steers a frozen generator using a per-candidate reward penalty, reward-gated feature diversity, and late-stage latent replacement. Each prompt returns four candidates.

## Run

Use Linux, Python 3.12, CUDA, and separate generation and reward environments. Model weights must already be cached; accept any upstream model-access conditions before downloading them. No model weights, private data, or cluster launchers are included.

```bash
# No model imports or GPU needed for this configuration check:
python generate.py --model sd15 --output outputs/example --dry-run

# After setting up the environments and caching the weights:
export SATISDIVE_REWARD_PYTHON=/absolute/path/to/reward-env/bin/python
CUDA_VISIBLE_DEVICES=0 python generate.py --model sd15 --output outputs/example

# FLUX generation uses logical GPU 0 and HPSv3 uses logical GPU 1:
CUDA_VISIBLE_DEVICES=0,1 python generate.py --model flux --output outputs/flux
```

Use `--prompts your_prompts.jsonl` for custom prompts, `--delta` for the reward tolerance, and `--seed` for the starting seed. JSONL records have a unique numeric string `prompt_id` and a nonempty `prompt`; see `example.jsonl`. The per-prompt seed is `seed + int(prompt_id)`. Output directories must be new, preventing accidental overwrites or mixing configurations. `run.json` records the resolved settings, reward-package versions, and prompt-file hash. FLUX writes `ours/<prompt_id>/sample_*.png`, SD1.5 writes `ours/<prompt_id>/candidate_*.png`, and SANA writes `ours/images/<prompt_id>/candidate_*.png`; each prompt directory also contains `diag.json`.

`SATISDIVE_WORKER_GPU` optionally selects a logical reward-device index within `CUDA_VISIBLE_DEVICES`: default 0 for SD1.5 and 1 for SANA/FLUX. FLUX/HPSv3 requires two GPUs; SANA also defaults to separate generation and reward GPUs. Memory requirements depend on the model and software environment.

## Fixed settings

| Preset | Generator / reward | Resolution | Steps / sampler | Scoring steps | Default tolerance |
| --- | --- | --- | --- | --- | --- |
| `flux` | FLUX.1-dev / HPSv3 | 1024 square | 28 Euler + marginal-preserving SDE | 4,8,12,16,20,24 | 1.5 |
| `sana` | SANA-1.6B / ImageReward | 1024 square | 20 Euler + marginal-preserving SDE | 3,6,9,12,15,18 | 0.5 |
| `sd15` | SD1.5 / ImageReward | 512 square | 100 DDIM, eta=1 | 20,40,60,80,99 | 0.75 |

Complete settings are in `presets/`, including fixed penalty weights, update size, feature choice, and replacement settings. Only the tolerance is exposed as a method control. `tau_relmax_offset` is the paper's Delta, not its diversity cutoff delta or evaluation floor rho. Null values select first-scoring-step batch calibration. The reward ramp and late-stage latent replacement are enabled. Features come from FLUX block 12 image-token attention, SANA block 7's `attn1` output, or the SD1.5 UNet mid-block. DreamSim is an evaluation metric, not the generation objective.

## Environments and weights

Use `requirements.txt` for generation and either `requirements-imagereward.txt` or `requirements-hpsv3.txt` for the separate reward environment. Do not combine them: the Transformers versions differ. Torch/torchvision must be compatible with the GPU driver.

For FLUX, install the included differentiable HPSv3 source into its reward environment with `pip install --no-deps ./third_party/HPSv3`; the HPSv3 requirements include the training-module dependencies imported by its inference API. SANA and SD1.5 use ImageReward. The runner does not install packages. Example setup for SD1.5/SANA, from this directory:

```bash
python3.12 -m venv .venv-generation
.venv-generation/bin/python -m pip install -r requirements.txt
python3.12 -m venv .venv-imagereward
.venv-imagereward/bin/python -m pip install -r requirements-imagereward.txt
export SATISDIVE_REWARD_PYTHON="$PWD/.venv-imagereward/bin/python"
.venv-generation/bin/python check_environment.py sd15
.venv-generation/bin/python generate.py --model sd15 --output outputs/example
```

For FLUX, create a separate `.venv-hpsv3`, install `requirements-hpsv3.txt` and the bundled HPSv3 source there, set `SATISDIVE_REWARD_PYTHON` to its `bin/python`, and use `check_environment.py flux`. The environment check loads no model weights. Generation also checks reward imports before loading the generator. Keep the venv executable path; do not replace it with the symlink's system-Python target.

Generator model IDs are `black-forest-labs/FLUX.1-dev`, `Efficient-Large-Model/Sana_1600M_1024px_diffusers`, and `stable-diffusion-v1-5/stable-diffusion-v1-5`. Reward dependencies include ImageReward-v1.0 and, for HPSv3, `MizzenAI/HPSv3` and Qwen2-VL-7B-Instruct. The inherited loaders do not pin model revisions; record the cached revisions when reproducing results. Reward libraries may fetch their assets on first use unless those assets are cached or offline mode is enabled.

## Scope and checks

This generation-only package includes fixed presets for FLUX, SANA, and SD1.5; baselines, the toy study, benchmark prompt redistribution, and evaluation scripts are not included.

Validation includes interface tests, module imports, and CPU checks of the sampling helpers. Fresh-install GPU execution of this packaged release has not yet been validated.

Run `python -m unittest discover -s tests` for the lightweight interface tests. Third-party source and attribution are identified in `THIRD_PARTY.md`.
