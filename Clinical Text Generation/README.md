# Clinical Text Generation and Interpretability with nanoGPT

Adapting a character-level GPT transformer (nanoGPT) for healthcare applications, trained on
MIMIC-III discharge summaries. The project implements the full pipeline — data loading, a
transformer built from scratch, training, generation, quantitative evaluation, attention-based
interpretability, and a comparison against Microsoft's BioGPT.

> ⚠️ **This is a research/educational project.** The model produces plausible-looking but not
> factually reliable clinical text and **must not be used for clinical decision-making**.

## Contents

| File | Description |
|------|-------------|
| `clinical_text_generation.ipynb` | End-to-end notebook (data → model → training → evaluation → interpretability) |
| `slides.pdf` | Presentation slides summarizing the project |
| `requirements.txt` | Python dependencies |

## Pipeline Overview

1. **Data Acquisition** — Load MIMIC-III discharge summaries from Google BigQuery
   (`physionet-data.mimiciii_notes.noteevents`), with local caching.
2. **Preprocessing** — Strip de-identification PII patterns (`[**...**]`), normalize whitespace.
3. **Tokenization** — Character-level tokenizer with explicit `<UNK>` handling.
4. **Model** — GPT-style transformer built from scratch: causal multi-head self-attention,
   pre-norm residual blocks, GELU feed-forward, token + position embeddings.
5. **Training** — AdamW, learning-rate scheduling, periodic train/val loss estimation,
   best-checkpoint saving, and loss-curve visualization.
6. **Generation** — Sampling with temperature and top-k controls.
7. **Evaluation** — Perplexity, vocabulary coverage, medical-term frequency, type-token ratio.
8. **Interpretability** — Attention-map visualization across transformer layers.
9. **Comparison** — 3-way comparison: trained nanoGPT vs. an untrained baseline vs. BioGPT (347M).

## Model Configuration

| Parameter | Value |
|-----------|-------|
| block_size | 256 |
| n_embd | 384 |
| n_head | 6 |
| n_layer | 6 |
| dropout | 0.2 |
| Parameters | ~10M |

## Getting Started

The notebook is designed to run in **Google Colab** (GPU runtime recommended).

1. Open `clinical_text_generation.ipynb` in Colab or Jupyter.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your Google Cloud project ID. The notebook contains a placeholder — replace
   `YOUR_GCP_PROJECT_ID` with your own project that has BigQuery access to the MIMIC-III dataset.
4. Run the cells top to bottom.

## Data Access & Compliance

This project uses **MIMIC-III**, a credentialed dataset hosted on
[PhysioNet](https://physionet.org/content/mimiciii/). Access requires:

- Completing the required CITI "Data or Specimens Only Research" training.
- Signing the PhysioNet Credentialed Health Data Use Agreement (DUA).
- BigQuery access to `physionet-data.mimiciii_notes`.

**No raw patient data is included in this repository.** Notebook outputs have been stripped, in
keeping with the MIMIC-III DUA, which prohibits redistributing the underlying clinical text. You
must obtain your own credentialed access to reproduce the results.

## Key Findings

- A ~10M-parameter character-level transformer can learn clinical language structure — medical
  terminology, section headers, and note formatting.
- The trained model substantially outperforms an untrained baseline across all metrics.
- BioGPT (347M, pretrained on biomedical literature) produces more coherent, grounded text, but
  the small character-level model offers finer-grained interpretability via attention maps.

## Limitations

- Character-level tokenization needs longer sequences than subword models for the same content.
- ~10M parameters is small compared to modern clinical NLP models.
- No factual grounding — outputs are not clinically reliable.
- Trained on a single document type (discharge summaries).

## Future Work

- Subword tokenization (BPE / SentencePiece).
- Larger models with gradient accumulation and mixed precision.
- Multi-task training across note types (radiology, progress notes, etc.).
- Clinical NER integration for structured extraction.
- Comparison with clinical LLMs (Med-PaLM, ClinicalBERT, GatorTron).

## Acknowledgements

- Built on the [nanoGPT](https://github.com/karpathy/nanoGPT) architecture by Andrej Karpathy.
- [MIMIC-III](https://physionet.org/content/mimiciii/) clinical database (Johnson et al., 2016).
- [BioGPT](https://github.com/microsoft/BioGPT) by Microsoft Research.

## Citations

This project uses the MIMIC-III Clinical Database, accessed via PhysioNet. If you use this work
or reproduce it, please cite the following.

**MIMIC-III database:**

> Johnson, A., Pollard, T., & Mark, R. (2016). MIMIC-III Clinical Database (version 1.4).
> PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/C2XW26

**Original MIMIC-III publication:**

> Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. H., Feng, M., Ghassemi, M., Moody, B.,
> Szolovits, P., Celi, L. A., & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care
> database. *Scientific Data*, 3, 160035.

**PhysioNet:**

> Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley,
> H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource
> for complex physiologic signals. *Circulation* [Online]. 101 (23), pp. e215–e220.
> RRID:SCR_007345.
