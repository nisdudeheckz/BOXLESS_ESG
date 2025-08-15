"""
Fit calibration from a CSV with columns: image_path,text,final_score,label
label: 1 if the chosen crop is correct, else 0
"""
import argparse, csv, os
from src.boxless_esg.calibration import fit_platt, fit_isotonic

def read_csv(path):
    scores, labels = [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            scores.append(float(row["final_score"]))
            labels.append(int(row["label"]))
    return scores, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--kind", choices=["platt","isotonic"], default="platt")
    args = ap.parse_args()

    scores, labels = read_csv(args.csv)
    if args.kind == "platt":
        model = fit_platt(scores, labels)
    else:
        model = fit_isotonic(scores, labels)
    model.save(args.out)
    print("Saved:", os.path.abspath(args.out))

if __name__ == "__main__":
    main()
