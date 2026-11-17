"""Thin wrapper around the lerobot pi0.5 policy (load + pre/post processors).

Adapted from the `Pi05Adapter` in action-atlas
(https://github.com/CWRU-AISM/action-atlas, Apache License 2.0),
trimmed to the pieces PilotVLA needs: model loading, the lerobot
pre/post-processor pipeline, and video saving.
"""
import os
from pathlib import Path


def _resolve_checkpoint(checkpoint: str) -> str:
    # Absolute path if it exists locally, else treat as a HuggingFace repo ID
    p = Path(checkpoint)
    if p.exists():
        return str(p.resolve())
    return checkpoint


class Pi05Adapter:
    name = "pi05"

    def __init__(self):
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.env_preprocessor = None
        self.env_postprocessor = None

    def load_model(self, checkpoint="lerobot/pi05_libero_finetuned", device="cuda"):
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
        os.environ.setdefault("PYTORCH_COMPILE_DISABLE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("MUJOCO_GL", "egl")
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors

        self.device = device
        checkpoint = _resolve_checkpoint(checkpoint)
        self.policy = PI05Policy.from_pretrained(checkpoint)
        self.policy.eval().to(device)
        self.policy.config.device = device
        self.model = self.policy
        self._checkpoint = checkpoint

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        return self.policy

    def _ensure_env_processors(self, suite, max_steps):
        if self.env_preprocessor is not None:
            return
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        env_cfg = LiberoEnvConfig(task=suite, observation_height=256, observation_width=256)
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=self.policy.config,
        )


def get_adapter(model_name: str) -> Pi05Adapter:
    if model_name != "pi05":
        raise ValueError(f"This release only ships the pi05 adapter (got: {model_name})")
    return Pi05Adapter()


def save_video(frames: list, path, fps: int = 10):
    if not frames:
        return
    try:
        import imageio
        writer = imageio.get_writer(
            str(path), fps=fps, codec="libx264", pixelformat="yuv420p",
            output_params=["-crf", "18", "-preset", "fast"],
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    except Exception as e:
        print(f"Warning: Could not save video {path}: {e}")
