import argparse
import sys
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Convert a pickled CTNet checkpoint to a pure state_dict .pth")
    p.add_argument("--in", dest="in_path", type=Path, required=True, help="Input .pth (pickled model or dict)")
    p.add_argument("--out", dest="out_path", type=Path, required=True, help="Output .pth (state_dict only)")
    p.add_argument("--device", type=str, default="cpu", help="map_location device for torch.load")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Import model classes and alias them under __main__ for legacy pickles
    from CTNet_model import (
        EEGTransformer,
        BranchEEGNetTransformer,
        PatchEmbeddingCNN,
        PositioinalEncoding,
        TransformerEncoder,
        TransformerEncoderBlock,
        MultiHeadAttention,
        FeedForwardBlock,
        ResidualAdd,
        ClassificationHead,
    )
    safe_types = [
        EEGTransformer,
        BranchEEGNetTransformer,
        PatchEmbeddingCNN,
        PositioinalEncoding,
        TransformerEncoder,
        TransformerEncoderBlock,
        MultiHeadAttention,
        FeedForwardBlock,
        ResidualAdd,
        ClassificationHead,
    ]

    try:
        from torch.serialization import safe_globals
    except Exception:
        safe_globals = None

    def alias_main():
        main_mod = sys.modules.setdefault("__main__", sys.modules.get("__main__"))
        for cls in safe_types:
            setattr(main_mod, cls.__name__, cls)

    alias_main()

    obj = None
    if safe_globals is not None:
        try:
            with safe_globals(safe_types):
                obj = torch.load(args.in_path, map_location=device, weights_only=True)
        except Exception:
            obj = None
    if obj is None:
        try:
            obj = torch.load(args.in_path, map_location=device, weights_only=False)
        except TypeError:
            obj = torch.load(args.in_path, map_location=device)

    # Extract a plain state_dict
    if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
        sd = obj.state_dict()
    elif isinstance(obj, dict):
        for k in ["state_dict", "model_state", "model", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                break
        else:
            sd = obj
    else:
        raise RuntimeError("Unrecognized checkpoint format; cannot extract state_dict")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, args.out_path)
    print(f"Saved state_dict to: {args.out_path}")


if __name__ == "__main__":
    main()

