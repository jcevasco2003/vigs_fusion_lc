#!/usr/bin/env python3
import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract NetVLAD descriptor with hloc")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--hloc_root", required=True, help="Path to Hierarchical-Localization root")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.hloc_root not in sys.path:
        sys.path.append(args.hloc_root)

    try:
        import torch
    except Exception as exc:
        sys.stderr.write(f"failed to import torch: {exc}\n")
        return 10

    try:
        from PIL import Image
    except Exception as exc:
        sys.stderr.write(f"failed to import Pillow (PIL): {exc}\n")
        return 11

    try:
        from hloc.extractors.netvlad import NetVLAD
    except Exception as exc:
        sys.stderr.write(f"failed to import hloc NetVLAD: {exc}\n")
        return 3

    try:
        pil_image = Image.open(args.image).convert("RGB")
    except Exception as exc:
        sys.stderr.write(f"failed to read input image: {exc}\n")
        return 1

    width, height = pil_image.size
    rgb_bytes = pil_image.tobytes()
    image = torch.ByteTensor(torch.ByteStorage.from_buffer(rgb_bytes))
    image = image.view(height, width, 3).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    conf = {
        "output": "global-feats-netvlad",
        "model": {"name": "netvlad"},
        "preprocessing": {"resize_max": 1024},
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = NetVLAD(conf).to(device).eval()
    except Exception as exc:
        sys.stderr.write(f"failed to initialize NetVLAD model: {exc}\n")
        return 4

    try:
        with torch.no_grad():
            descriptor = model({"image": image.to(device)})["global_descriptor"]
    except Exception as exc:
        sys.stderr.write(f"failed during forward pass: {exc}\n")
        return 5

    desc = descriptor.detach().cpu().reshape(-1)
    if int(desc.numel()) != 4096:
        sys.stderr.write(f"unexpected descriptor size {int(desc.numel())}\n")
        return 2

    out = " ".join(f"{float(v):.8g}" for v in desc.tolist())
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
