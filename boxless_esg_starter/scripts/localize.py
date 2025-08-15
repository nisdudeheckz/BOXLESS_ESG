# --- begin bootstrap for src/ layout ---
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
# --- end bootstrap ---
import argparse

from boxless_esg.pipeline import run_pipeline

def main():
    ap = argparse.ArgumentParser(description="Boxless ESG — minimal scaffold")
    ap.add_argument("--image", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    run_pipeline(args.image, args.text, args.out_dir)
    print("Done.")

if __name__ == "__main__":
    main()
