# LLMs in Healthcare: Drug Synergy & Diabetes Risk Prediction

This project explores using Large Language Models (LLMs) for real-world healthcare tasks — specifically **drug combination synergy prediction** and **diabetes risk prediction** — demonstrating that LLMs already encode massive medical knowledge and can accelerate research that would otherwise take years of custom model development.

> **Key Insight**: A custom ML model trained from scratch for diabetes prediction achieved <50% accuracy. Using Claude via Amazon Bedrock with just 4 patient examples (few-shot) hit **83.3% accuracy** — no training, no feature engineering, no months of iteration. These foundation models can seriously accelerate healthcare research and benefit human civilization at scale.
>
> We need to stop reducing these models to chatbot duties — crafting emails, skimming spreadsheets, writing replies. That's using a rocket ship to deliver pizza. The real power is in research, prediction, and solving problems that would take humans years. Let's aim higher. 🚀

---

## Notebooks

### 1. `ChatGPT_API.ipynb` — Drug Synergy Prediction (OpenAI)

Uses OpenAI's ChatGPT API to predict whether drug combinations are synergistic in cancer cell lines. Designed for Google Colab.

**Sections:**
1. **Setup & Data Loading** — Install OpenAI, load the 718K drug synergy dataset
2. **Zero-Shot Prompting** — Ask GPT-4 to predict synergy with no examples
3. **Chat-Based Approach** — System prompts + multi-turn conversations for richer context
4. **Batch Evaluation** — Evaluate on endometrium tissue test data (14 samples)
5. **OpenAI Embeddings + Classifier** — `text-embedding-ada-002` embeddings → Logistic Regression
6. **CancerGPT** — Fine-tuned GPT-2 model for synergy classification

**Models Used:** GPT-4, text-embedding-ada-002, CancerGPT (fine-tuned GPT-2)

---

### 2. `Bedrock_Claude_API.ipynb` — Drug Synergy Prediction (AWS Bedrock)

Same drug synergy workflow as above, but using **Claude on Amazon Bedrock** — no OpenAI key needed, uses AWS credentials instead.

**Sections:**
1. **Setup & Data Loading** — boto3 + Bedrock client, load synergy dataset
2. **Zero-Shot Claude Prompting** — Single predictions, system prompts, multi-turn conversations
3. **Batch Evaluation** — Claude Haiku on endometrium test data
4. **Titan Embeddings + Classifier** — Amazon Titan Embeddings → Logistic Regression
5. **Visualization** — ROC curve, Precision-Recall curve, Confusion Matrix, Model Comparison

**Models Used:** Claude Sonnet 4.6, Claude Haiku 4.5, Amazon Titan Embed Text v1

**Drug Synergy Results (Endometrium Tissue):**

| Method | AUROC | AUPRC |
|--------|-------|-------|
| Claude Zero-Shot (Haiku) | 0.29 | 0.50 |
| Titan Embeddings + Logistic Regression | **0.86** | **0.75** |

---

### 3. `LLM_Healthcare_Tutorial.ipynb` — Diabetes Risk Prediction (AWS Bedrock)

Predicts diabetes risk from synthetic patient records (Synthea) using multiple LLM prompting strategies. Demonstrates how different prompting approaches dramatically affect prediction quality.

**Sections:**
1. **Setup & Data Loading** — Load Synthea patient data (117 patients), build patient profiles from conditions, observations, and medications
2. **Zero-Shot Prompting** — Predict diabetes risk with no examples
3. **Few-Shot In-Context Learning** — Provide 4 labeled patient examples before prediction
4. **Chain-of-Thought Reasoning** — Step-by-step clinical reasoning before prediction
5. **Medical Record Summarization** — Summarize records for doctors vs. patients (plain language)
6. **Evaluation & Comparison** — Side-by-side accuracy comparison with visualization
7. **Ideas for Improvement** — Ensemble methods, fine-tuning, richer profiles, Tree-of-Thought

**Models Used:** Claude Sonnet 4.6, Claude Haiku 4.5

**Diabetes Prediction Results (30 test patients):**

| Prompting Method | Accuracy | Notes |
|-----------------|----------|-------|
| Zero-Shot | 66.7% | No examples, just the question |
| Few-Shot (4 examples) | **83.3%** | 2 positive + 2 negative examples |
| Chain-of-Thought | 46.7% | Over-reasoning led to worse predictions |

**Key Takeaway:** Simple few-shot prompting with just 4 examples massively outperformed both zero-shot and chain-of-thought. More reasoning ≠ better predictions for structured clinical tasks.

---

## Datasets

| File | Description |
|------|-------------|
| `data_synergy.csv` | 718K drug combination records with synergy scores (Loewe model) |
| `train_test_split.json` | Pre-defined train/test splits by tissue type |
| `cancergpt.pt` | Pre-trained CancerGPT model weights (fine-tuned GPT-2) |
| `synthea_sample_data_csv_latest/` | Synthea synthetic patient data — patients, conditions, observations, medications, etc. |
| `10k_synthea_covid19_csv/` | 10K Synthea COVID-19 synthetic patient dataset |

---

## Setup

### For ChatGPT Notebook (Google Colab)
```bash
pip install openai
```
Requires an OpenAI API key.

### For Bedrock Notebooks (Local)
```bash
pip install boto3 pandas numpy scikit-learn tqdm matplotlib
```
```bash
ada credentials update --account=<YOUR_ACCOUNT> --provider=conduit --role=<YOUR_ROLE>
```
Requires AWS account with Bedrock model access enabled for Claude and Titan in `us-east-1`.

---

## Project Structure

```
LLM healthcare/
├── ChatGPT_API.ipynb                  # Drug synergy — OpenAI (Colab)
├── Bedrock_Claude_API.ipynb           # Drug synergy — AWS Bedrock
├── LLM_Healthcare_Tutorial.ipynb      # Diabetes prediction — AWS Bedrock
├── data_synergy.csv                   # Drug synergy dataset (718K records)
├── train_test_split.json              # Train/test splits
├── cancergpt.pt                       # CancerGPT model weights
├── synthea_sample_data_csv_latest/    # Synthea patient data
├── 10k_synthea_covid19_csv/           # COVID-19 Synthea data
├── generate_slides.py                 # Slide generation script
└── README.md                          # This file
```

---

## Why This Matters

These experiments show that LLMs aren't just chatbots — they're research accelerators. Building a custom diabetes prediction model from scratch took significant effort and couldn't break 50% accuracy. Meanwhile, Claude with 4 examples hit 83.3%.

The same pattern holds for drug synergy: embedding-based approaches using foundation models (Titan Embeddings) achieved 0.86 AUROC, dramatically outperforming zero-shot prompting.

Foundation models already encode vast medical knowledge from their training. The right prompting strategy unlocks it — and few-shot learning is often all you need.
