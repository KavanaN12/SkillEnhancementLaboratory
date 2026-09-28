from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"


def clean_text(text):
    """Basic preprocessing of the academic report."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(text, tokenizer, max_tokens=900):
    """Split a long report into manageable token-sized chunks."""
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(
            tokenizer.encode(sentence, add_special_tokens=False)
        )

        if current_chunk and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def summarize_text(text, tokenizer, model):
    """Generate a summary using the pre-trained Hugging Face model."""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=130,
            min_length=40,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )


def main():

    input_file = "academic_report.txt"
    output_file = "summary.txt"

    print("Loading Hugging Face model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    with open(input_file, "r", encoding="utf-8") as file:
        report = file.read()

    report = clean_text(report)

    print("Processing academic report...")

    chunks = split_into_chunks(
        report,
        tokenizer
    )

    print(f"Report divided into {len(chunks)} chunk(s).")

    summaries = []

    for i, chunk in enumerate(chunks, start=1):

        print(f"Summarizing chunk {i}/{len(chunks)}...")

        summary = summarize_text(
            chunk,
            tokenizer,
            model
        )

        summaries.append(summary)

    combined_summary = " ".join(summaries)

    # If the report was divided into multiple chunks,
    # summarize the combined intermediate summaries.
    if len(summaries) > 1:

        print("Generating final concise summary...")

        final_summary = summarize_text(
            combined_summary,
            tokenizer,
            model
        )

    else:
        final_summary = combined_summary

    print("\n" + "=" * 70)
    print("GENERATED SUMMARY")
    print("=" * 70)

    print(final_summary)

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(final_summary)

    print("\nSummary saved to:", output_file)


if __name__ == "__main__":
    main()