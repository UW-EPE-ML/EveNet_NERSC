# Ablation Study for Pretraining

> Only 20M model and 500M dataset are used for ablation study.

There are 4 cases to compare:

# Extra Models (Ablation Runs)

| Model Variant                                            | Classification | EventGen | TruthGen | Segmentation | Notes                                 |
|----------------------------------------------------------|----------------|----------|----------|--------------|---------------------------------------|
| **Baseline**                                             | ✅              | ✅        | ✅        | ✅            | Full multi-task training              |
| **1. [Fully Unsupervised](pretrain-yulei-20M-1.yaml)**   | ❌              | ✅        | ❌        | ❌            | Pure self-supervised (event gen only) |
| **2. Generation Only**                                   | ❌              | ✅        | ✅        | ❌            | Generative tasks only                 |
| **3. Deterministic Only**                                | ✅              | ❌        | ❌        | ❌            | Classification only                   |
| **4. [Without Segmentation](pretrain-yulei-20M-4.yaml)** | ✅              | ✅        | ✅        | ❌            | Test effect of removing segmentation  |
