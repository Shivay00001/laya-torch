"""Export a standalone MLX checkpoint. Weight downloads stay outside Git."""

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from .agent import Agent


def convert(
    model_id_or_path, output, *, dtype="float16", revision=None, subfolder=None, token=None
):
    """Convert an upstream checkpoint to MLX parameter names and the requested dtype.

    The destination must not exist. If export fails, the newly created partial
    directory is removed; existing checkpoints are never overwritten.
    """
    output = Path(output).expanduser()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    agent = Agent(
        model_id_or_path, dtype=dtype, revision=revision, subfolder=subfolder, token=token
    )
    output.mkdir(parents=True)
    try:
        shutil.copytree(agent.model_dir / "tokenizer", output / "tokenizer")
        (output / "encoder").mkdir()
        (output / "encoder/config.json").write_text(json.dumps(agent.encoder_cfg, indent=2) + "\n")
        (output / "rl_agent_config.json").write_text(json.dumps(agent.cfg, indent=2) + "\n")
        save_file(
            agent.model.state_dict(), str(output / "model.safetensors")
        )
        metadata = {
            "format": "laya-mlx",
            "format_version": 1,
            "dtype": dtype,
            "source": str(model_id_or_path),
            "revision": revision,
            "subfolder": subfolder,
        }
        (output / "mlx_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    return output
