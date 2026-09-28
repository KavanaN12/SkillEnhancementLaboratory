"""
Pretrained Word Embeddings Laboratory
-------------------------------------

Tasks:
1. Explore pretrained GloVe word vectors and perform vector arithmetic.
2. Use PCA to visualize 10 technology-domain word embeddings.
3. Generate the 5 most semantically similar words for a given input.

Pretrained Model:
    GloVe Wiki-Gigaword 100-dimensional vectors

Run:
    python word_embeddings_lab.py

Run semantic similarity for a specific word:
    python word_embeddings_lab.py --word computer
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gensim.downloader import load
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_NAME = "glove-wiki-gigaword-100"

OUTPUT_DIR = Path("outputs")

# Ten words selected from the technology domain.
TECH_WORDS = [
    "computer",
    "software",
    "hardware",
    "internet",
    "network",
    "server",
    "database",
    "programming",
    "algorithm",
    "security",
]


# ---------------------------------------------------------------------
# Load pretrained embeddings
# ---------------------------------------------------------------------

def load_embeddings():
    print(f"Loading pretrained model: {MODEL_NAME}")
    print("The first run may take some time because the model is downloaded.")

    model = load(MODEL_NAME)

    print(
        f"Loaded {len(model.key_to_index):,} words "
        f"with {model.vector_size} dimensions."
    )

    return model


# ---------------------------------------------------------------------
# Vector arithmetic
# ---------------------------------------------------------------------

def vector_arithmetic(model, word_a, word_b, word_c, topn=5):
    """
    Perform:

        vector(A) - vector(B) + vector(C)

    and find the nearest words to the resulting vector.
    """

    # Explicit vector arithmetic.
    result_vector = (
        model[word_a]
        - model[word_b]
        + model[word_c]
    )

    # Find words closest to the resulting vector.
    results = model.similar_by_vector(
        result_vector,
        topn=topn + 10
    )

    # Remove the original input words from the results.
    excluded = {word_a, word_b, word_c}

    filtered_results = [
        (word, score)
        for word, score in results
        if word not in excluded
    ]

    return filtered_results[:topn]


def run_vector_arithmetic(model):
    print("\n" + "=" * 70)
    print("PART 1: VECTOR ARITHMETIC")
    print("=" * 70)

    # Classic word relationships provide clear demonstrations
    # of semantic relationships in the embedding space.
    examples = [
        ("king", "man", "woman"),
        ("paris", "france", "italy"),
        ("walk", "walking", "swim"),
    ]

    results = []

    for word_a, word_b, word_c in examples:

        expression = f"{word_a} - {word_b} + {word_c}"

        matches = vector_arithmetic(
            model,
            word_a,
            word_b,
            word_c,
            topn=5
        )

        print(f"\n{expression}")

        for rank, (word, score) in enumerate(matches, 1):
            print(
                f"  {rank}. {word:<20} "
                f"cosine similarity = {score:.4f}"
            )

        results.append(
            (
                expression,
                matches
            )
        )

    return results


# ---------------------------------------------------------------------
# PCA visualization
# ---------------------------------------------------------------------

def visualize_pca(model):

    print("\n" + "=" * 70)
    print("PART 2: PCA VISUALIZATION")
    print("=" * 70)

    available_words = [
        word for word in TECH_WORDS
        if word in model.key_to_index
    ]

    missing_words = [
        word for word in TECH_WORDS
        if word not in model.key_to_index
    ]

    if missing_words:
        print(
            "Warning: The following words were not found:",
            ", ".join(missing_words)
        )

    # Extract 100-dimensional vectors.
    vectors = np.array([
        model[word]
        for word in available_words
    ])

    # Reduce 100 dimensions to 2 dimensions.
    pca = PCA(
        n_components=2,
        random_state=42
    )

    reduced_vectors = pca.fit_transform(vectors)

    explained_variance = pca.explained_variance_ratio_

    print(
        f"PCA explained variance: "
        f"PC1 = {explained_variance[0] * 100:.2f}%, "
        f"PC2 = {explained_variance[1] * 100:.2f}%"
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    output_path = (
        OUTPUT_DIR /
        "technology_embeddings_pca.png"
    )

    # -------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------

    plt.figure(figsize=(11, 7))

    plt.scatter(
        reduced_vectors[:, 0],
        reduced_vectors[:, 1],
        s=100
    )

    for word, x, y in zip(
        available_words,
        reduced_vectors[:, 0],
        reduced_vectors[:, 1]
    ):
        plt.annotate(
            word,
            (x, y),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=10
        )

    plt.title(
        "PCA Visualization of Technology-Domain Word Embeddings"
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Saved visualization to: "
        f"{output_path.resolve()}"
    )

    # -------------------------------------------------------------
    # Pairwise cosine similarity
    # -------------------------------------------------------------

    normalized_vectors = (
        vectors /
        np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True
        )
    )

    similarity_matrix = (
        normalized_vectors @ normalized_vectors.T
    )

    pairs = []

    for i in range(len(available_words)):

        for j in range(i + 1, len(available_words)):

            pairs.append(
                (
                    similarity_matrix[i, j],
                    available_words[i],
                    available_words[j]
                )
            )

    pairs.sort(reverse=True)

    print("\nClosest pairs among the 10 selected words:")

    for score, word1, word2 in pairs[:5]:

        print(
            f"  {word1:<15} <-> "
            f"{word2:<15}: {score:.4f}"
        )

    return (
        available_words,
        reduced_vectors,
        explained_variance,
        output_path,
        pairs
    )


# ---------------------------------------------------------------------
# Semantic similarity
# ---------------------------------------------------------------------

def semantic_similarity(model, input_word):

    print("\n" + "=" * 70)
    print("PART 3: SEMANTIC SIMILARITY")
    print("=" * 70)

    input_word = input_word.lower().strip()

    if input_word not in model.key_to_index:

        print(
            f"\n'{input_word}' is not present "
            f"in the pretrained vocabulary."
        )

        print(
            "Try a common word such as:"
            " computer, software, network, security"
        )

        return []

    results = model.most_similar(
        input_word,
        topn=5
    )

    print(
        f"\nFive semantically similar words "
        f"for '{input_word}':"
    )

    for rank, (word, score) in enumerate(
        results,
        1
    ):

        print(
            f"  {rank}. {word:<20} "
            f"cosine similarity = {score:.4f}"
        )

    return results


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------

def save_results(
    arithmetic_results,
    similarity_results,
    input_word,
    explained_variance,
    closest_pairs
):

    OUTPUT_DIR.mkdir(exist_ok=True)

    summary_path = (
        OUTPUT_DIR /
        "results_summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "PRETRAINED WORD EMBEDDINGS LAB RESULTS\n"
        )

        file.write("=" * 65 + "\n\n")

        # ---------------------------------------------------------
        # Model
        # ---------------------------------------------------------

        file.write("PRETRAINED MODEL\n")
        file.write(
            f"{MODEL_NAME}\n"
        )

        file.write(
            "Embedding dimension: 100\n\n"
        )

        # ---------------------------------------------------------
        # Vector arithmetic
        # ---------------------------------------------------------

        file.write(
            "VECTOR ARITHMETIC\n"
        )

        file.write(
            "Formula: vector(A) - vector(B) + vector(C)\n\n"
        )

        for expression, matches in arithmetic_results:

            file.write(
                f"{expression}\n"
            )

            for rank, (word, score) in enumerate(
                matches,
                1
            ):

                file.write(
                    f"{rank}. {word} "
                    f"({score:.4f})\n"
                )

            file.write("\n")

        # ---------------------------------------------------------
        # PCA
        # ---------------------------------------------------------

        file.write(
            "PCA EXPLAINED VARIANCE\n"
        )

        file.write(
            f"PC1: "
            f"{explained_variance[0] * 100:.2f}%\n"
        )

        file.write(
            f"PC2: "
            f"{explained_variance[1] * 100:.2f}%\n\n"
        )

        # ---------------------------------------------------------
        # Closest technology word pairs
        # ---------------------------------------------------------

        file.write(
            "CLOSEST TECHNOLOGY WORD PAIRS\n"
        )

        for score, word1, word2 in closest_pairs[:5]:

            file.write(
                f"{word1} <-> {word2}: "
                f"{score:.4f}\n"
            )

        file.write("\n")

        # ---------------------------------------------------------
        # Semantic similarity
        # ---------------------------------------------------------

        file.write(
            "SEMANTIC SIMILARITY\n"
        )

        file.write(
            f"Input word: {input_word}\n"
        )

        for rank, (word, score) in enumerate(
            similarity_results,
            1
        ):

            file.write(
                f"{rank}. {word} "
                f"({score:.4f})\n"
            )

    print(
        f"\nSaved result summary to: "
        f"{summary_path.resolve()}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Explore pretrained word embeddings "
            "using GloVe."
        )
    )

    parser.add_argument(
        "--word",
        default="computer",
        help=(
            "Input word for semantic similarity "
            "(default: computer)"
        )
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    # Load pretrained embeddings.
    model = load_embeddings()

    # Part 1.
    arithmetic_results = (
        run_vector_arithmetic(model)
    )

    # Part 2.
    (
        words,
        reduced_vectors,
        explained_variance,
        output_path,
        closest_pairs
    ) = visualize_pca(model)

    # Part 3.
    similarity_results = semantic_similarity(
        model,
        args.word
    )

    # Save all important results.
    save_results(
        arithmetic_results,
        similarity_results,
        args.word,
        explained_variance,
        closest_pairs
    )

    print(
        "\nExperiment completed successfully."
    )


if __name__ == "__main__":
    main()