Boxless ESG: Scene Localization Pipeline

This repository contains an optimized scene-localization pipeline that extends Grounding DINO with custom creativity and interpretability layers. The goal is to detect and localize objects or relations in natural scenes without relying only on bounding boxes.

🚀 Features

Hybrid Proposals: MobileSAM + filtered Selective Search (≤50, deduped).

Hierarchical Scoring: Lightweight GDINO gate + heavy scoring (CLIP, GDINO overlap, Relation, Affordance).

Overlay Support: Save query overlays on images for visualization.

Negative Test Handling: Supports queries that do not exist in the image.

Dataset Included: Test images are included under data/demo_images/.

📂 Repository Structure
boxless_esg_starter/
│
├── scripts/
│   ├── run_one.py           # Main entry for running single queries
│   ├── run_batch.py         # Run multiple queries on multiple images
│
├── src/boxless_esg/
│   ├── pipeline.py          # Core pipeline logic
│   ├── overlay.py           # Overlay saving utility
│   ├── scoring/             # CLIP, GDINO, Relation, Affordance modules
│   ├── proposals/           # MobileSAM + Selective Search hybrid
│
├── weights/                 # Pretrained model weights (add via release / Git LFS)
│
├── data/
│   ├── demo_images/         # Sample images for testing
│   │   ├── OIP (1).jpeg
│   │   ├── OIP (2).jpeg
│   │   ├── OIP.jpeg
│
├── runs/                    # Outputs saved here
│
└── README.md

⚙️ Installation
git clone https://github.com/<your-username>/<your-repo>.git
cd boxless_esg_starter

# (Optional but recommended)
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

pip install -r requirements.txt

▶️ Usage
Single Image + Query
python -m scripts.run_one \
  --image "data/demo_images/OIP (1).jpeg" \
  --text "a person in a white t shirt" \
  --out_dir runs/test1 \
  --max_regions 600 \
  --gate_keep 8 \
  --save_overlay

Example Test Cases (x4)

data/demo_images/OIP (1).jpeg → "a person in a white t shirt"

data/demo_images/OIP (1).jpeg → "a person using phone"

data/demo_images/OIP.jpeg → "a child playing with a ball"

data/demo_images/OIP (2).jpeg → "a dog jumping over fence" (negative test)

📊 Output

Overlays and results are saved to runs/<exp_name>/

Each run saves:

overlay.png → image with highlighted detections

results.json → structured outputs with scores + evidence

📦 Weights

Model weights should be placed in the weights/ directory.
You can use Git LFS for managing large model files.

git lfs install
git lfs track "*.pth"

📌 Notes

The save_overlay feature has been integrated into the pipeline.

Works with both positive queries and negative test cases.

Extendable for new scoring modules and datasets.

📝 License

MIT License © 2025
