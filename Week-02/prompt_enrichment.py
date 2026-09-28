import os
import re
import json
import gensim.downloader as api

from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.7-flash"
EMBEDDING_MODEL = "glove-wiki-gigaword-100"

ORIGINAL_PROMPT = "Explain cybersecurity."

# Maximum number of embedding candidates to retrieve
TOP_N_SIMILAR = 15

# Number of useful words to add to the enriched prompt
MAX_ENRICHMENT_WORDS = 6


# ============================================================
# LOAD API KEY
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Check your .env file."
    )

client = genai.Client(api_key=api_key)


# ============================================================
# LOAD WORD EMBEDDING MODEL
# ============================================================

print("\nLoading word embedding model...")
print("This may take a little time on the first run.\n")

embedding_model = api.load(EMBEDDING_MODEL)

print("Embedding model loaded successfully!")


# ============================================================
# EXTRACT IMPORTANT WORD FROM PROMPT
# ============================================================

def extract_topic(prompt):
    """
    Extract the main topic from the example prompt.

    For this lab we use a simple keyword-based approach.
    """

    cybersecurity_keywords = [
        "cybersecurity",
        "cyber security",
        "network security",
        "information security",
        "computer security",
        "cybercrime"
    ]

    prompt_lower = prompt.lower()

    for keyword in cybersecurity_keywords:
        if keyword in prompt_lower:
            return keyword.replace(" ", "_")

    # Fallback:
    words = re.findall(r"[a-zA-Z]+", prompt_lower)

    stop_words = {
        "explain",
        "describe",
        "define",
        "what",
        "is",
        "are",
        "the",
        "a",
        "an",
        "about",
        "of",
        "and",
        "in",
        "on"
    }

    useful_words = [
        word for word in words
        if word not in stop_words
    ]

    if useful_words:
        return useful_words[0]

    return "security"


# ============================================================
# GET SIMILAR WORDS USING WORD EMBEDDINGS
# ============================================================

def get_similar_words(topic, top_n=TOP_N_SIMILAR):

    # Some multi-word topics are not single entries
    topic = topic.replace(" ", "_")

    if topic not in embedding_model.key_to_index:

        # Try individual words if the complete topic is unavailable
        candidates = []

        for word in topic.split("_"):
            if word in embedding_model.key_to_index:
                candidates.extend(
                    embedding_model.most_similar(
                        word,
                        topn=top_n
                    )
                )

        return candidates[:top_n]

    return embedding_model.most_similar(
        topic,
        topn=top_n
    )


# ============================================================
# FILTER EMBEDDING RESULTS
# ============================================================

def filter_relevant_words(similar_words):

    """
    GloVe is a general-purpose embedding model.
    Some retrieved words may be semantically related but
    not useful for our cybersecurity prompt.

    We therefore keep terms that belong to a cybersecurity
    vocabulary.
    """

    cybersecurity_terms = {
        "security",
        "cybersecurity",
        "cybercrime",
        "cyberattack",
        "cyberattacks",
        "attack",
        "attacks",
        "threat",
        "threats",
        "vulnerability",
        "vulnerabilities",
        "malware",
        "ransomware",
        "phishing",
        "privacy",
        "encryption",
        "authentication",
        "authorization",
        "network",
        "networks",
        "firewall",
        "firewalls",
        "protection",
        "defense",
        "defence",
        "intrusion",
        "intrusion-detection",
        "detection",
        "prevention",
        "data",
        "information",
        "computer",
        "computers",
        "digital",
        "online",
        "internet",
        "systems",
        "system",
        "preparedness",
        "informatics"
    }

    selected = []

    for word, score in similar_words:

        clean_word = word.lower().replace("_", "-")

        if clean_word in cybersecurity_terms:
            selected.append((word, score))

    # If the embedding model does not return enough
    # domain-specific terms, add reliable related terms.
    fallback_terms = [
        "threat",
        "vulnerability",
        "cybercrime",
        "encryption",
        "authentication",
        "privacy"
    ]

    existing_words = {
        word.lower() for word, _ in selected
    }

    for word in fallback_terms:

        if len(selected) >= MAX_ENRICHMENT_WORDS:
            break

        if word not in existing_words:
            selected.append((word, 0.0))
            existing_words.add(word)

    return selected[:MAX_ENRICHMENT_WORDS]


# ============================================================
# CREATE ENRICHED PROMPT
# ============================================================

def create_enriched_prompt(original_prompt, selected_words):

    words = [word for word, _ in selected_words]

    word_list = ", ".join(words)

    enriched_prompt = f"""
{original_prompt}

Expand the explanation by covering these related concepts:
{word_list}

Provide:
1. A clear definition.
2. Explanation of the major concepts.
3. Practical examples.
4. How these concepts contribute to protecting systems and data.

Keep the response focused and relevant to the original topic.
"""

    return enriched_prompt.strip()


# ============================================================
# GEMINI GENERATION
# ============================================================

def generate_response(prompt):

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text.strip()


# ============================================================
# BASIC RESPONSE STATISTICS
# ============================================================

def calculate_statistics(response, selected_words):

    words = re.findall(r"\b\w+\b", response.lower())

    word_count = len(words)

    unique_words = len(set(words))

    keywords = [
        word.lower()
        for word, _ in selected_words
    ]

    keywords_found = [
        word for word in keywords
        if word in response.lower()
    ]

    keyword_coverage = (
        len(keywords_found) / len(keywords) * 100
        if keywords
        else 0
    )

    return {
        "word_count": word_count,
        "unique_words": unique_words,
        "keyword_coverage": keyword_coverage,
        "keywords_found": keywords_found
    }


# ============================================================
# GEMINI-BASED COMPARISON
# ============================================================

def compare_responses(
    original_prompt,
    enriched_prompt,
    original_response,
    enriched_response
):

    comparison_prompt = f"""
You are evaluating a prompt-enrichment experiment.

Original prompt:
{original_prompt}

Enriched prompt:
{enriched_prompt}

Response generated from the original prompt:
{original_response}

Response generated from the enriched prompt:
{enriched_response}

Compare the two responses.

Score each response from 1 to 10 for:

1. Detail:
How thoroughly does the response explain the topic?

2. Relevance:
How directly does the response address the requested topic?

3. Specificity:
How concrete and focused is the response?

4. Topic Coverage:
How many important aspects of the topic are covered?

Return ONLY valid JSON in this format:

{{
    "original": {{
        "detail": 0,
        "relevance": 0,
        "specificity": 0,
        "topic_coverage": 0
    }},
    "enriched": {{
        "detail": 0,
        "relevance": 0,
        "specificity": 0,
        "topic_coverage": 0
    }},
    "overall_conclusion": "..."
}}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=comparison_prompt
    )

    text = response.text.strip()

    # Remove markdown code fences if Gemini adds them
    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:

        return {
            "original": {
                "detail": "N/A",
                "relevance": "N/A",
                "specificity": "N/A",
                "topic_coverage": "N/A"
            },
            "enriched": {
                "detail": "N/A",
                "relevance": "N/A",
                "specificity": "N/A",
                "topic_coverage": "N/A"
            },
            "overall_conclusion": text
        }


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(" WORD EMBEDDING BASED PROMPT ENRICHMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: Original prompt
    # --------------------------------------------------------

    print("\n[STEP 1] ORIGINAL PROMPT")
    print("-" * 70)
    print(ORIGINAL_PROMPT)

    # --------------------------------------------------------
    # STEP 2: Extract topic
    # --------------------------------------------------------

    topic = extract_topic(ORIGINAL_PROMPT)

    print("\n[STEP 2] EXTRACTED TOPIC")
    print("-" * 70)
    print(topic)

    # --------------------------------------------------------
    # STEP 3: Retrieve similar words
    # --------------------------------------------------------

    print("\n[STEP 3] WORD EMBEDDING RESULTS")
    print("-" * 70)

    similar_words = get_similar_words(topic)

    for rank, (word, score) in enumerate(
        similar_words,
        start=1
    ):
        print(
            f"{rank:2}. {word:20} "
            f"similarity = {score:.4f}"
        )

    # --------------------------------------------------------
    # STEP 4: Select relevant terms
    # --------------------------------------------------------

    selected_words = filter_relevant_words(
        similar_words
    )

    print("\n[STEP 4] SELECTED TERMS FOR ENRICHMENT")
    print("-" * 70)

    for word, score in selected_words:

        if score > 0:
            print(
                f"{word:20} "
                f"embedding score = {score:.4f}"
            )
        else:
            print(
                f"{word:20} "
                f"fallback domain term"
            )

    # --------------------------------------------------------
    # STEP 5: Create enriched prompt
    # --------------------------------------------------------

    enriched_prompt = create_enriched_prompt(
        ORIGINAL_PROMPT,
        selected_words
    )

    print("\n[STEP 5] ENRICHED PROMPT")
    print("-" * 70)
    print(enriched_prompt)

    # --------------------------------------------------------
    # STEP 6: Generate original response
    # --------------------------------------------------------

    print("\n[STEP 6] GENERATING ORIGINAL RESPONSE...")
    print("-" * 70)

    original_response = generate_response(
        ORIGINAL_PROMPT
    )

    print(original_response)

    # --------------------------------------------------------
    # STEP 7: Generate enriched response
    # --------------------------------------------------------

    print("\n[STEP 7] GENERATING ENRICHED RESPONSE...")
    print("-" * 70)

    enriched_response = generate_response(
        enriched_prompt
    )

    print(enriched_response)

    # --------------------------------------------------------
    # STEP 8: Basic statistics
    # --------------------------------------------------------

    original_stats = calculate_statistics(
        original_response,
        selected_words
    )

    enriched_stats = calculate_statistics(
        enriched_response,
        selected_words
    )

    print("\n[STEP 8] BASIC RESPONSE STATISTICS")
    print("-" * 70)

    print(
        f"{'Metric':25}"
        f"{'Original':15}"
        f"{'Enriched':15}"
    )

    print("-" * 55)

    print(
        f"{'Word count':25}"
        f"{original_stats['word_count']:<15}"
        f"{enriched_stats['word_count']:<15}"
    )

    print(
        f"{'Unique words':25}"
        f"{original_stats['unique_words']:<15}"
        f"{enriched_stats['unique_words']:<15}"
    )

    print(
        f"{'Keyword coverage':25}"
        f"{original_stats['keyword_coverage']:.1f}%"
        f"{'':5}"
        f"{enriched_stats['keyword_coverage']:.1f}%"
    )

    # --------------------------------------------------------
    # STEP 9: AI evaluation
    # --------------------------------------------------------

    print("\n[STEP 9] AI-BASED COMPARISON")
    print("-" * 70)
    print("Evaluating detail, relevance, specificity and coverage...\n")

    comparison = compare_responses(
        ORIGINAL_PROMPT,
        enriched_prompt,
        original_response,
        enriched_response
    )

    original_scores = comparison["original"]
    enriched_scores = comparison["enriched"]

    print(
        f"{'Criterion':25}"
        f"{'Original':15}"
        f"{'Enriched':15}"
    )

    print("-" * 55)

    criteria = [
        ("Detail", "detail"),
        ("Relevance", "relevance"),
        ("Specificity", "specificity"),
        ("Topic Coverage", "topic_coverage")
    ]

    for label, key in criteria:

        print(
            f"{label:25}"
            f"{str(original_scores[key]):<15}"
            f"{str(enriched_scores[key]):<15}"
        )

    print("\nOVERALL CONCLUSION")
    print("-" * 70)
    print(comparison["overall_conclusion"])

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(" EXPERIMENT COMPLETED")
    print("=" * 70)

    print("\nPipeline:")
    print("Original Prompt")
    print("      ↓")
    print("Word Embeddings")
    print("      ↓")
    print("Similar Words")
    print("      ↓")
    print("Relevant Term Selection")
    print("      ↓")
    print("Enriched Prompt")
    print("      ↓")
    print("Gemini Generative AI")
    print("      ↓")
    print("Original vs Enriched Response")
    print("      ↓")
    print("Comparison")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()