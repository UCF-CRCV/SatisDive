# Third-party components

The `third_party/HPSv3` directory contains the HPSv3 source needed by the differentiable reward worker. Its original MIT licence and upstream attribution are retained. Upstream: https://github.com/MizzenAI/HPSv3. The source includes a differentiable image-processing path. This package does not redistribute model weights.

`implementation/sd15/ddim.py` extracts the shared DDIM, classifier-free-guidance, and decode functions from this project's DAS adaptation. The DDIM mechanics follow Diffusers and the DAS implementation (https://github.com/krafton-ai/DAS). This file is a dependency of SatisDive, not an included DAS baseline experiment.

FLUX and SANA use the marginal-preserving flow SDE described by Mark et al., Feynman-Kac steering. The model adapters depend on Hugging Face Diffusers. ImageReward and all pretrained generators retain their upstream licences and access conditions. Third-party author names are attribution, not submission-author metadata.

The repository's root `LICENSE` contains the Apache License 2.0. Third-party components retain their respective licences and attribution as described above.
