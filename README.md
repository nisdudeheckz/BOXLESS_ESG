# Boxless ESG — Scene Localization Without Explicit Bounding Boxes

Boxless ESG is a modular, interpretable pipeline for **text-based scene localization** that works without explicit bounding boxes in training data.

It takes an **image** and a **natural language query**, and outputs:
- A **mask** of the detected region
- A **narrative crop** for storytelling
- An **evidence JSON** containing all intermediate scores and reasoning

---

## 🚀 Features
- **Open-vocabulary** — Works with arbitrary queries
- **Multi-stage fusion** — Combines CLIP semantics, GDINO gating, spatial relations, affordances
- **Explainable** — Outputs full scoring breakdown
- **No retraining required** — Works zero-shot with pretrained models

---

## 📂 Repository Structure
src/
boxless_esg/
pipeline.py # Main orchestration logic
gdino_gate.py # GDINO-based text-aware gating
clip_sem.py # CLIP semantic similarity
relations.py # Spatial reasoning functions
affordance.py # Affordance context scoring
precision_bbox.py # Special-case precision mode
utils.py # Box/mask utilities
proposals/
selective_search.py # Region proposal generation
docs/
final_documentation.md # Full 6-page project documentation
weights/
clip/ # CLIP pretrained model
gdino/ # GDINO pretrained model

yaml
Copy
Edit

---

## ⚙️ Installation

### 1️⃣ Clone the repo
```bash
git clone https://github.com/<your-username>/boxless-esg.git
cd boxless-esg
2️⃣ Create environment
bash
Copy
Edit
conda create -n boxless-esg python=3.10 -y
conda activate boxless-esg
3️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
📥 Pretrained Weights
CLIP (ViT-L/14) — Download

Grounding DINO — Download

Place weights in weights/ directory following:

bash
Copy
Edit
weights/
  clip/ViT-L-14.pt
  gdino/groundingdino_swinb_cogcoor.pth
▶️ Usage
Basic Command
bash
Copy
Edit
python -m src.boxless_esg \
  --image "test_images/cart_people.jpg" \
  --text "two people near a cart" \
  --out_dir outputs/demo1 \
  --max_regions 600 \
  --gate_keep_k 60
Output Files
mask.png – Binary mask of detected region

narrative_crop.jpg – Cropped region for storytelling

evidence.json – Scoring breakdown for interpretability

📊 Example Results
Query	Input Image	Output Crop
"Two people near a cart"	
"Person left of bicycle"	

🛠 Technical Overview
The pipeline:

Generates proposals with Selective Search

Deduplicates using IoU filtering

Passes proposals through GDINO gate to retain text-relevant candidates

Scores with CLIP for semantic match

Adds relation scoring for “near”, “left of”, “right of”

Penalizes oversized regions (area penalty)

Adds affordance reasoning (placeholder mode)

Fuses scores with weights: clip=0.70, gdino=0.05, rel=0.20, afford=0.05

Outputs the best match and full evidence

Full architecture details: Documentation


📧 Contact
For queries, contact:

 Nischal Deep

GitHub: nisdudecheckz
