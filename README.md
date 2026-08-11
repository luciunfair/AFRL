# AFRL: A Lightweight Statistical Framework for Real-Time Detection of Encrypted File Fragments

**Official implementation** of the AFRL framework — a fast, training-free method for telling encrypted data apart from other file content, one 4096-byte fragment at a time.

📄 Paper: *AFRL: A Lightweight Statistical Framework for Real-Time Detection of Encrypted File Fragments* — Alireza Aliaskari Hosseinabadi, Mehdi Teimouri ([SSRN preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6799741))

---

## What is AFRL?

Digital forensics and data-recovery tools often run into file fragments with no headers, no metadata, and no clue about the file system they came from. Figuring out whether such a fragment is **encrypted** or just some other kind of dense, unstructured data (compressed, multimedia, etc.) is a real problem — and doing it without decrypting anything or knowing the encryption algorithm in advance is harder still.

AFRL tackles this by combining four lightweight statistical tests instead of a trained model:

- **Approximate Entropy Test** (adapted from NIST SP 800-22)
- **Frequency (Monobit) Test within a Block** (adapted from NIST SP 800-22)
- **Runs Test** (adapted from NIST SP 800-22)
- **Local Nibble Variance (LNV)** — a new test introduced in this work, which measures how much the distribution of 4-bit nibbles fluctuates across sliding windows of a fragment. Structured data (e.g. compressed content) tends to show high local variance; encrypted data tends to look uniform.

A fragment is flagged as **encrypted** only if it passes all four tests, and as **non-encrypted** otherwise. Because there's no training step, AFRL sidesteps the dataset bias and overfitting risk that come with machine-learning and deep-learning detectors, and it generalizes well to fragment types it has never seen.

## Why it's useful

- **No training required** — no labeled datasets, no retraining, no hyperparameter search.
- **Fast** — processes a fragment in well under a millisecond, making it practical for real-time forensic pipelines and resource-constrained environments.
- **Accurate** — evaluated on the ITC-IT and EnCoD benchmark datasets (mixed fragments from compressed files, multimedia, text, and encrypted data):

  | Dataset | AFRL Accuracy |
  |---|---|
  | ITC-IT | 95.38% |
  | EnCoD | 84.09% |

  This beats the strongest existing statistical baseline by 1.87% (ITC-IT) and 1.63% (EnCoD), and outperforms the best machine-learning-based methods by 8.33% and 2.14% respectively.
- **Robust** — since it doesn't learn from a specific dataset, it holds up well on heterogeneous, out-of-distribution fragment types.

## Installation

```bash
git clone https://github.com/luciunfair/AFRL.git
cd AFRL
pip install -r requirements.txt
```

Requires Python 3.8+ and:

```
numpy==1.26.4
pandas==2.2.0
scipy==1.12.0
tqdm==4.66.1
```

## Usage

Run the script directly:

```bash
python AFRL.py
```

You'll be prompted for two paths:

1. **Input CSV** — a dataset where each row is a byte fragment written as a list of integers (0–255), e.g. `[234, 2, 91, ...]`. Each row is analyzed independently, so fragments don't need to be the same length (though the original evaluation uses 4096-byte fragments).
2. **Output CSV** — where the predicted labels are written, one per row, in the same order as the input: `0` = not encrypted, `1` = encrypted.

The script processes fragments in parallel across CPU cores (via `ProcessPoolExecutor`) and shows a progress bar while it runs, printing the total execution time when it finishes.

### Using it as a library

You can also import the core function directly in your own code:

```python
from AFRL import AFRL_test

fragment = [234, 2, 91, ...]  # a byte fragment as a list/array of ints (0-255)
is_encrypted = AFRL_test(fragment)  # True if encrypted, False otherwise
```

## How it works

Each fragment goes through all four tests, and each test returns pass/fail based on how consistent the fragment's statistical properties are with what's expected from truly random (i.e., encrypted) data:

1. **Approximate Entropy** checks whether short bit patterns occur about as often as expected in random data.
2. **Frequency Within Block** checks whether the proportion of 1-bits stays close to 0.5 across blocks of the bitstream.
3. **Runs Test** checks whether the number of consecutive runs of identical bits matches what's expected for random data.
4. **Local Nibble Variance (LNV)** slides a window across the fragment, computes a chi-square statistic on the 4-bit nibble distribution in each window, and checks whether the variance of those scores stays below a threshold — low variance points to encryption, high variance points to structured (non-random) content.

A fragment only gets labeled *encrypted* when all four agree.



## License

Released under the [MIT License](LICENSE).

## Contact

Alireza Aliaskari Hosseinabadi — [alireza.aliaskari76@gmail.com](mailto:alireza.aliaskari76@gmail.com)
