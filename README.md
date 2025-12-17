# The Cyborg's Pen: Quantifying the Emotional Gap and Stylistic Divergence in Human vs. AI Narratives

This repository contains the dataset, source code, and experimental results for the research paper **"The Cyborg's Pen"** (submitted to ACL).

## 📄 Abstract

As Large Language Models (LLMs) increasingly enter the domain of creative writing, the line between human and machine storytelling is blurring. In this study, we analyze a parallel corpus of over **8,600 narratives** to examine the 'emotional gap' between human authors and AI models, from standard baselines (GPT-4o-mini) to advanced systems like **Claude 3.5 Sonnet** and **Gemini 3 Pro**.

We find distinct problems: although SOTA models have mastered human-level vocabulary, they suffer from a systemic 'emotional bias' stemming from safety alignment (RLHF). These AI narratives tend to force positivity and avoid realistic conflict at the expense of narrative depth. Our results show that while this surface-level mimicry can fool traditional TF-IDF detectors, deep semantic classifiers (DistilBERT) can still expose the underlying emotional sterility of machine-generated text.

## 📂 Repository Structure

The project is organized to reproduce the three Research Questions (RQs) from the paper:

- **`data/`**: 
  - `final_dataset_human_vs_ai.csv`: The core parallel corpus containing prompts and stories from Humans, GPT-4o-mini, Gemini Flash, Claude Haiku, Claude Sonnet, and Gemini Pro.
  - `writingprompts.csv`: Preprocessed prompts source.
  - **`outputs/`**: Contains generated figures (PNG), metric tables (CSV), and topic visualization (HTML).

- **`scripts/`**: 
  - **Data Collection**:
    - `collect_data.py`: Fetches AI narratives via OpenRouter API.
    - `preprocess.py`: Cleans raw WritingPrompts data.
  - **RQ1: Stylistic & Emotional Analysis**:
    - `analyze_q1.py`: Extracts linguistic features (TTR, POS) and emotional metrics (VADER, NRC, Empath).
    - `analyze_emotional_thematic_link.py`: Links authorship to negative emotion density.
  - **RQ2: Semantic & Thematic Analysis**:
    - `analyze_q2.py`: Performs BERTopic modeling and semantic similarity checks.
    - `plot_q2_revised.py`: Generates the aggregated thematic distribution chart (Figure 3).
    - `q2_generate_appendix.py`: Generates semantic heatmaps and keyword tables.
  - **RQ3: Attribution & Detection**:
    - `analyze_q3.py`: Fine-tunes DistilBERT for detection.
    - `analyze_q3_tfidf.py`: Runs the TF-IDF baseline.
    - `analyze_q3_comparative.py`: Compares "Hidden AI" vs. "Obvious AI".
    - `analyze_q3_attribution.py`: Visualizes the gap attribution (Figure 5).

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* GPU recommended for BERTopic and DistilBERT training (CUDA enabled).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Vargram1/Cyborgs-Pen.git](https://github.com/Vargram1/Cyborgs-Pen.git)
    cd Cyborgs-Pen
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup API Keys (for Data Collection only):**
    Create a file named `keys.py` in the `scripts/` folder and add your OpenRouter key:
    ```python
    OPENROUTER_API_KEY = "your_key_here"
    ```

## 📊 Reproduction Steps

To reproduce the figures and tables from the paper:

1.  **Run Feature Extraction (RQ1):**
    ```bash
    python scripts/analyze_q1.py
    ```
2.  **Run Topic Modeling (RQ2):**
    ```bash
    python scripts/analyze_q2.py
    python scripts/plot_q2_revised.py
    ```
3.  **Run Classification Experiments (RQ3):**
    ```bash
    python scripts/analyze_q3_tfidf.py
    python scripts/analyze_q3.py
    python scripts/analyze_q3_attribution.py
    ```

All results will be saved to `data/outputs/`.

## 📜 Citation

If you use this dataset or code, please cite our paper:

```bibtex
@inproceedings{zhou2025cyborgspen,
  title={The Cyborg's Pen: Quantifying the Emotional Gap and Stylistic Divergence in Human vs. AI Narratives},
  author={Tianji Zhou},
  booktitle={Proceedings of the Association for Computational Linguistics (ACL)},
  year={2025}
}