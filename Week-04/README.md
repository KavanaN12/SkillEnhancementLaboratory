# Academic Report Summarizer

This project uses a pre-trained Hugging Face Transformer model to generate a concise summary from a long academic report.

## Technologies

- Python
- Hugging Face Transformers
- PyTorch
- Pre-trained model: `sshleifer/distilbart-cnn-12-6`

## Project Files

- `summarizer.py` - main implementation
- `academic_report.txt` - sample academic report
- `requirements.txt` - required Python packages
- `summary.txt` - generated automatically after execution

## Setup

### 1. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the program

```bash
python summarizer.py
```

The first execution downloads the pre-trained Hugging Face model. Internet access is required for the first run.

## Using another academic report

Replace the contents of `academic_report.txt` with your own report and run:

```bash
python summarizer.py
```

The generated summary is displayed in the terminal and saved to `summary.txt`.

## Working

1. The academic report is loaded from a text file.
2. Basic text preprocessing removes unnecessary whitespace.
3. A tokenizer checks the length of the input.
4. Long reports are divided into manageable chunks.
5. Each chunk is summarized using the pre-trained DistilBART model.
6. The intermediate summaries are combined.
7. If necessary, the combined summaries are summarized again to produce the final concise summary.
8. The final summary is displayed and saved in `summary.txt`.

## Note

The model is pre-trained and is used directly for abstractive text summarization. No model training or custom dataset is required.
