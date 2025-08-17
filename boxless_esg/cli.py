from __future__ import annotations
import argparse, os, sys
from .pipeline import run_pipeline

def main(argv=None):
    ap = argparse.ArgumentParser(prog="boxless-esg", description="Boxless ESG: sub-scene localization")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--text", required=True, help="Query text")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--max_regions", type=int, default=200)
    ap.add_argument("--gate_keep_k", type=int, default=20)
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    result = run_pipeline(
        image_path=args.image,
        text_query=args.text,
        out_dir=args.out_dir,
        max_regions=args.max_regions,
        gate_keep_k=args.gate_keep_k,
    )
    print("Saved:", result["crop_path"])
    print("       ", result["mask_path"])
    print("       ", result["evidence_path"])

if __name__ == "__main__":
    main()
