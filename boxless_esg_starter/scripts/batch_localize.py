# scripts/batch_localize.py
import csv, os, argparse, json
from boxless_esg.pipeline import run_pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True, help="CSV with columns: image,text")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--csv_out", default="runs_batch.csv")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    rows_out = []
    with open(args.pairs_csv, newline="") as f:
        rdr = csv.DictReader(f)
        for i,row in enumerate(rdr, start=1):
            img, text = row["image"], row["text"]
            out_dir = os.path.join(args.out_root, f"ex_{i:04d}")
            os.makedirs(out_dir, exist_ok=True)
            res = run_pipeline(img, text, out_dir)
            with open(res["evidence_path"]) as ef:
                ev = json.load(ef)
            rows_out.append({
                "image_path": img,
                "text": text,
                "final_score": ev["winner"]["final_score"],
                "prob_calibrated": ev["winner"].get("prob_calibrated", ""),
                "out_dir": out_dir,
            })
            print(f"[{i}] done:", out_dir)

    with open(args.csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print("Wrote:", os.path.abspath(args.csv_out))

if __name__ == "__main__":
    main()
