# Pretrained Word Embeddings Laboratory

## Objective

This experiment explores pretrained word embeddings and demonstrates three major applications:

1. Performing vector arithmetic using pretrained word vectors.
2. Reducing high-dimensional word embeddings using PCA and visualizing them.
3. Finding five semantically similar words for a given input.

The technology domain is used for the visualization task.

## Technologies Used

- Python
- Gensim
- NumPy
- Scikit-learn
- Matplotlib
- GloVe pretrained word embeddings

## Pretrained Model

The experiment uses:

`glove-wiki-gigaword-100`

The model contains 400,000 words represented using 100-dimensional vectors.

The first execution downloads the pretrained model automatically through Gensim.

## Project Structure

```text
Week-05/
│
├── word_embeddings_lab.py
├── requirements.txt
├── README.md
├── analysis_notes.txt
│
└── outputs/
    ├── technology_embeddings_pca.png
    └── results_summary.txt