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
