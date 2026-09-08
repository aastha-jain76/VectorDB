# External Requirements & Download Checklist

## Project: Custom Vector Database Engine from Scratch
**Status**: Ready for Verification  
**File**: `docs/requirements_checklist.md`

---

## 1. Executive Summary: What You Need to Download

Good news: **Most of the required Python libraries are ALREADY installed in your environment**!
We have inspected your system:
- **Operating System**: Linux (x86_64)
- **CPU**: 11th Gen Intel(R) Core(TM) i5-1135G7 (8 threads)
- **Disk Space**: 95 GB free
- **Python**: 3.13.9
- **Pre-installed**: `numpy` (2.5.2), `torch` (2.11.0+cpu), `sentence-transformers` (5.4.1), `streamlit` (1.51.0), `scipy` (1.16.3).

The only items that need to be acquired from the internet or prepared are:
1. **The Embedding Model Weights** (~80 MB, downloaded once and cached locally).
2. **A Real Text Corpus** (to provide realistic text queries and semantic matches for the demo).

Below is the exact checklist.

---

## 2. Detailed Requirements Checklist

### Item 1: Embedding Model (Automatic or Offline)
To transform text sentences into real 384-dimensional vectors for the vector database:
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Size**: ~80 MB.
- **Action Required by You**:
  - **If Internet is available (Default)**: **Nothing!** Our automated preparation script (`python3 data/prepare_data.py`) will automatically download and cache it locally in `~/.cache/huggingface/`.
  - **If completely Offline**: You can choose the synthetic vector generator (which generates 50,000 clustered vectors with NumPy without any internet), OR download `all-MiniLM-L6-v2` weights from another machine and copy them into `models/all-MiniLM-L6-v2`.

### Item 2: Text Corpus (5,000 to 50,000 texts)
The problem statement allows two options:
> *"Embed a real text corpus (any 5,000 short texts you like) or generate clustered synthetic vectors from your SEED."*

You have three choices:
- **Choice A: Automated Public Dataset (Recommended)**:
  - Download a clean public text dataset (e.g., AG News or DBpedia subsets, ~10 MB).
  - Our data prep script will download this automatically via HTTP.
- **Choice B: Your Own Custom Text Files**:
  - If you have specific text files, PDFs, or FAQs you want to search in your demo video, simply drop a `.txt` or `.jsonl` file into `data/corpus/`.
- **Choice C: Pure Synthetic Clustered Vectors (100% Offline)**:
  - Requires 0 external downloads. Generates 50,000 anisotropic clustered vectors using NumPy Gaussian mixtures with `SEED = 42`.

### Item 3: Screen Recorder / Video Tool (For Demo Submission)
The problem statement asks to *"upload a working demo video of it"*:
- Recommended Linux screen recorders (choose any if not already installed):
  - **OBS Studio**: `sudo apt install obs-studio`
  - **SimpleScreenRecorder**: `sudo apt install simplescreenrecorder`
  - **Kazam**: `sudo apt install kazam`
  - Built-in GNOME screen recorder (press `Ctrl + Alt + Shift + R` or `Print Screen` -> Video icon).

---

## 3. Checklist Table

| Item | Source / Command | Size | Status | Action Needed |
| :--- | :--- | :--- | :--- | :--- |
| **Python 3.10+** | System | - | Installed (Python 3.13) | None |
| **NumPy (Linear Algebra)** | `pip install numpy` | ~20 MB | Installed (2.5.2) | None |
| **PyTorch (CPU)** | `pip install torch` | ~180 MB | Installed (2.11.0) | None |
| **Sentence-Transformers** | `pip install sentence-transformers` | ~15 MB | Installed (5.4.1) | None |
| **Streamlit (Demo UI)** | `pip install streamlit` | ~10 MB | Installed (1.51.0) | None |
| **Matplotlib (Charts)** | `pip install matplotlib` | ~30 MB | Installed | None |
| **MiniLM Embedding Model** | Hugging Face Hub | ~80 MB | Auto-download on run | None (Handled by script) |
| **50k Text Corpus** | Hugging Face / Wikipedia subset | ~12 MB | Auto-download on run | None (Handled by script) |
| **Screen Recorder** | GNOME / Kazam / OBS | - | Optional for video | Ensure you have one ready |

---

## 4. Next Step
Once you review this checklist, run:
```bash
python3 data/prepare_data.py
```
This will automatically verify all assets, download the required lightweight model & corpus, compute and cache the 50,000 embeddings, and precompute the 500 ground truth queries.
