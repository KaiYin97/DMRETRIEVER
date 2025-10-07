# DMRETRIEVER
# DMRetriever: A Family of Models for Improved Text Retrieval in Disaster Management

## 📖 Introduction

Effective and efficient access to accurate and relevant information is essential for disaster management. However, existing general-domain models fail to achieve consistent state-of-the-art (SOTA) performance across diverse search intents.

To this end, we introduce **DMRetriever**, a family of dense retrieval models tailored for disaster management. DMRetriever is trained through a novel three-stage framework:

1. **Bidirectional Attention Adaptation**  
2. **Unsupervised Contrastive Pre-training**  
3. **Difficulty-Aware Progressive Instruction Fine-tuning**

Training leverages high-quality data generated through an advanced **data refinement pipeline**:
1. **Domain-specific data synthesis**
2. **Mutual-agreement-based filtering**
3. **Difficulty-aware hard negative mining**

Comprehensive experiments on the **DisastIR benchmark** (the only public Information Retrieval benchmark for disaster management) demonstrate that DMRetriever achieves **SOTA performance across all six disaster-related search intents at every model scale**. Moreover, DMRetriever is highly **parameter-efficient**:  
- The **596M model** outperforms baselines over **13.3× larger**.  
- The **33M model** exceeds baselines with only **7.6% of their parameters**.

We will release the code, model checkpoints, training data, and evaluation scripts to facilitate future research. Please stay tuned!

---

## 🚀 Model Family

DMRetriever is released in multiple scales to support different deployment scenarios:

- **Small** (33M / 109M) — lightweight models for resource-constrained environments  
- **Medium** (335M) — balanced accuracy and efficiency  
- **Large** (596M / 1.5B) — strong performance with higher capacity  
- **XL** (4B / 7.6B) — best overall performance across all tasks  

---

## 📊 Benchmark Results on DisastIR-Test

| Model | Scale | QA | QAdoc | TW | FC | NLI | STS | Avg. |
|-------|-------|----|-------|----|----|-----|-----|------|
| **Small Size (≤109M)** 
| thenlper-gte-small | 33M | 18.04 | 9.13 | 10.95 | 49.63 | 37.51 | 55.55 | 30.14 |
| arctic-embed-m | 109M | 33.15 | 14.04 | 8.48 | 35.07 | 38.67 | 56.20 | 30.94 |
| thenlper-gte-base | 109M | 9.18 | 5.42 | 37.91 | 60.45 | 42.52 | 46.07 | 33.59 |
| arctic-embed-m-v1.5 | 109M | 25.76 | 30.41 | 17.95 | 47.97 | 42.88 | 64.16 | 38.19 |
| arctic-embed-s | 33M | 38.58 | 28.81 | 21.33 | 47.21 | 39.85 | 66.96 | 40.46 |
| bge-small-en-v1.5 | 33M | 56.91 | 51.19 | 25.15 | 55.17 | 32.87 | 64.54 | 47.64 |
| bge-base-en-v1.5 | 109M | 51.50 | 52.78 | 46.72 | 59.93 | 41.16 | 68.63 | 53.45 |
| **DMRetriever-33M (ours)** | 33M | 62.47 | 57.03 | 57.22 | 60.81 | 46.56 | 68.00 | 58.68 |
| **DMRetriever-109M (ours)** | 109M | 63.19 | 59.55 | 58.88 | 62.48 | 46.93 | 68.79 | 59.97 |
| **Medium Size (137M–335M)** |||||||||
| arctic-embed-m-long | 137M | 21.51 | 10.86 | 19.24 | 36.13 | 41.67 | 54.94 | 30.73 |
| arctic-embed-l | 335M | 40.56 | 30.19 | 14.98 | 32.64 | 34.20 | 56.10 | 34.78 |
| bge-large-en-v1.5 | 335M | 56.76 | 54.45 | 32.20 | 54.90 | 35.11 | 64.47 | 49.65 |
| gte-base-en-v1.5 | 137M | 60.52 | 55.63 | 46.24 | 52.23 | 39.62 | 70.41 | 54.11 |
| mxbai-embed-large-v1 | 335M | 64.24 | 62.63 | 39.94 | 58.12 | 40.18 | 68.01 | 55.52 |
| arctic-embed-m-v2.0 | 305M | 61.22 | 62.20 | 47.01 | 57.79 | 42.29 | 64.51 | 55.84 |
| **DMRetriever-335M (ours)** | 335M | 67.44 | 62.69 | 62.16 | 64.42 | 49.69 | 70.71 | 62.85 |
| **Large Size (560M–1.5B)** |||||||||
| arctic-embed-l-v2.0 | 568M | 55.23 | 59.11 | 38.11 | 60.10 | 41.07 | 62.61 | 52.70 |
| gte-large-en-v1.5 | 434M | 67.40 | 58.24 | 39.44 | 52.66 | 34.50 | 66.43 | 53.11 |
| Qwen3-Embedding-0.6B | 596M | 66.10 | 52.31 | 62.38 | 64.89 | 50.30 | 67.39 | 60.56 |
| multilingual-e5-large-instruct | 560M | 67.97 | 64.64 | 62.25 | 66.78 | 48.51 | 63.42 | 62.26 |
| multilingual-e5-large | 560M | 66.99 | 64.01 | 62.81 | 59.87 | 50.93 | 74.12 | 63.12 |
| gte-Qwen2-1.5B-instruct | 1.5B | 69.85 | 59.17 | 65.09 | 62.73 | 55.51 | 73.58 | 64.32 |
| inf-retriever-v1-1.5b | 1.5B | 69.41 | 64.29 | 62.99 | 65.39 | 54.03 | 73.92 | 65.01 |
| **DMRetriever-596M (ours)** | 596M | 72.44 | 67.50 | 65.79 | 69.15 | 55.71 | 74.73 | 67.55 |
| **XL Size (≥4B)** |||||||||
| Qwen3-Embedding-8B | 7.6B | 44.21 | 34.38 | 41.56 | 42.04 | 32.53 | 42.95 | 39.61 |
| gte-Qwen2-7B-instruct | 7.6B | 70.24 | 47.41 | 63.08 | 31.62 | 53.71 | 74.88 | 56.82 |
| NV-Embed-v1 | 7.9B | 68.06 | 62.70 | 56.02 | 59.64 | 48.05 | 67.06 | 60.26 |
| Qwen3-Embedding-4B | 4B | 67.20 | 59.14 | 65.28 | 67.16 | 53.61 | 58.51 | 61.82 |
| e5-mistral-7b-instruct | 7.1B | 65.57 | 64.97 | 63.31 | 67.86 | 47.55 | 66.48 | 62.58 |
| NV-Embed-v2 | 7.9B | 74.47 | 69.37 | 42.40 | 68.32 | 58.20 | 76.07 | 64.80 |
| inf-retriever-v1 | 7.1B | 72.84 | 66.74 | 66.23 | 65.53 | 51.86 | 75.98 | 66.53 |
| SFR-Embedding-Mistral | 7.1B | 71.41 | 67.14 | 69.45 | 70.31 | 50.93 | 72.67 | 66.99 |
| Linq-Embed-Mistral | 7.1B | 74.40 | 70.31 | 64.11 | 70.64 | 52.46 | 71.25 | 67.19 |
| **DMRetriever-4B (ours)** | 4B | 75.32 | 70.23 | 70.55 | 71.44 | 57.63 | 77.38 | 70.42 |
| **DMRetriever-7.6B (ours)** | 7.6B | 76.19 | 71.27 | 71.11 | 72.47 | 58.81 | 78.36 | 71.37 |

---

## Technical Details
The technical details for DMRETRIEVER will be released soon. Please stay tuned!

---

## Citation
TBA


