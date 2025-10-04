"""Load torch.nn.Module parameters from checkpoint."""

from pathlib import Path
from typing import Any

import torch
from torch import nn


def _remove_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Remove common prefix if present in state_dict parameters."""
    print(f"Removing prefix '{prefix}' from state_dict keys.")
    return {key.removeprefix(prefix): value for key, value in state_dict.items()}


def load_model(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load model parameters from checkpoint.

    Args:
        model (nn.Module): The model to load parameters into.
        checkpoint_path (Path): Path to the model checkpoint file.
        device (torch.device, optional): Device to map the model to. Defaults to CPU.

    Returns:
        nn.Module: The model with loaded parameters.
    """
    print(f"Loading pretrained model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle nested state dicts
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = _remove_prefix(state_dict, "module.")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"⚠️ Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    model.to(device)
    return model
