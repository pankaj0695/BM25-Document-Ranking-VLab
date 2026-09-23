"""
Virtual Laboratory Experiment: BM25 Based Document Ranking (Streamlit)
Implements the BM25 probabilistic ranking function from first principles and
compares its ranking behaviour against the classic TF-IDF vector-space model
(cosine similarity) on a shared document corpus and query.

Structure follows the IIT Kharagpur Virtual Labs sidebar convention:
  1. Purpose: The aim and learning objectives of the experiment.
  2. Theory: Concepts (TF-IDF, BM25), algorithm pipeline diagram, applications, procedure, key terms.
  3. Simulation: Interactive corpus/query/parameter controls, dual ranking engine,
     comparison plots, term-contribution breakdown, behind-the-scenes visualizations, and trial logger.
  4. Quiz: Self-grading conceptual assessment with instant feedback.
  5. Report Generation: Student info, recorded trials, observations, and downloadable PDF report.
  6. Certificate: Personalized PDF certificate, gated on completing the simulation + quiz.
  7. References: Citations with direct links to the source papers.

UI: a light theme (.streamlit/config.toml) plus a small injected stylesheet give a fixed, non-scrolling
sidebar, card-based pages and flicker-free navigation. The Simulation page embeds a client-side
animation of the retrieval pipeline that plays automatically.
"""

import io
import re
import json
import html as _html
import math
import collections
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


# ======================================================================================
# 0. GLOSSARY (hover definitions for key terms)
# ======================================================================================

GLOSSARY = {
    "TF-IDF": "Term Frequency times Inverse Document Frequency — a classic vector-space scoring "
              "formula that weights each term by how often it appears in a document and how rare "
              "it is across the collection.",
    "BM25": "Best Matching 25 — a probabilistic ranking function that improves on TF-IDF with "
            "term-frequency saturation and document-length normalization.",
    "TF": "Term Frequency — the number of times a term appears in a single document.",
    "IDF": "Inverse Document Frequency — a measure of how rare a term is across the whole "
           "document collection.",
    "cosine similarity": "The cosine of the angle between two vectors; measures how similar "
                          "their direction is, independent of their magnitude.",
    "k1": "BM25 parameter that controls how quickly the contribution of a repeated term "
          "saturates (diminishing returns).",
    "b": "BM25 parameter that controls how strongly a document's length is penalized relative "
         "to the collection average.",
    "avgdl": "The average document length (in tokens) across the entire corpus.",
    "saturate": "The effect where repeating a term more and more adds progressively less to the "
                "score, instead of growing without limit.",
    "vector space model": "A model that represents documents and queries as vectors of term "
                           "weights in a shared multi-dimensional space.",
    "tokenization": "The process of splitting raw text into individual words (tokens) for "
                     "processing.",
    "stopwords": "Very common words (like 'the', 'is', 'and') that are filtered out before "
                 "scoring because they carry little meaning.",
    "embedding": "A numeric vector representation of a document or query, positioned in a "
                 "shared space so that similar items land close together.",
    "rank correlation": "A statistic (Spearman's rho) measuring how similarly two different "
                         "rankings order the same set of items.",
    "document-length normalization": "Adjusting a document's score based on whether it is "
                                      "longer or shorter than the collection average, so long "
                                      "documents don't win purely by being long.",
}


def gterm(label: str, key: str = None) -> str:
    """Wraps `label` in an HTML <abbr> tag so hovering over it shows a short glossary definition."""
    definition = GLOSSARY.get(key or label, "").replace('"', "&quot;")
    return (
        f'<abbr title="{definition}" '
        f'style="text-decoration:underline dotted; text-underline-offset:3px; cursor:help;">'
        f'{label}</abbr>'
    )


# ======================================================================================
# 1. EXPERIMENT CONFIGURATION & EDUCATIONAL CONTENT
# ======================================================================================

EXPERIMENT_CONFIG = {
    "title": "BM25 Based Document Ranking",
    "objectives": [
        "Understand the TF-IDF vector-space model and cosine similarity as a baseline document ranking method.",
        "Implement the BM25 probabilistic ranking function, including its term-frequency saturation (k1) and "
        "document-length normalization (b) components.",
        "Empirically compare BM25 and TF-IDF rankings on the same corpus and query.",
        "Analyze how tuning k1 and b changes ranking outcomes, especially for long or keyword-repetitive documents."
    ]
}

THEORY_CONTENT = {
    "background": f"""
### Overview & Objective
Given a collection of documents and a user query, a ranking function assigns each document a
relevance score so the most relevant documents can be shown first. This experiment implements
two such functions — **{gterm('TF-IDF')} with {gterm('cosine similarity')}** and **{gterm('BM25')}**
— and lets you compare how they rank the same documents for the same query.

### TF-IDF Vector Space Model
Each document (and the query) is represented as a vector over the vocabulary (a {gterm('vector space model', 'vector space model')}).
Every term's weight combines **{gterm('Term Frequency (TF)', 'TF')}** — how often the term occurs in that document — with
**{gterm('Inverse Document Frequency (IDF)', 'IDF')}** — how rare the term is across the whole collection:

> weight(t, d) = tf(t, d) × idf(t),  where idf(t) = log(N / df(t)) + 1

Relevance is then measured as the **{gterm('cosine similarity')}** between the query vector and each document
vector — the cosine of the angle between them, independent of document length. A key limitation of
the classic (raw-count) TF-IDF formulation used here is that term frequency has **no saturation**:
repeating a query term many more times keeps increasing a document's score with no diminishing returns.

### BM25 Ranking Function
{gterm('BM25')} ("Best Matching 25") is a probabilistic ranking function that scores a document D against a
query Q as:

> score(D, Q) = Σ IDF(qᵢ) · [ f(qᵢ, D) · (k1 + 1) ] / [ f(qᵢ, D) + k1 · (1 − b + b · |D| / avgdl) ]

where f(qᵢ, D) is the frequency of query term qᵢ in D, |D| is the document's length in tokens, and
{gterm('avgdl')} is the average document length across the collection. Two tunable parameters shape the score:

- **{gterm('k1', 'k1')}** (typically 1.2 – 2.0) controls how quickly additional occurrences of a term **{gterm('saturate', 'saturate')}** —
  beyond a point, repeating a term contributes progressively less to the score.
- **{gterm('b', 'b')}** (0 to 1) controls **{gterm('document-length normalization', 'document-length normalization')}** — how much a document's score is
  penalized for being longer than the collection average (b = 1: full normalization, b = 0: none).

### Why BM25 Often Ranks Better than Plain TF-IDF
Because BM25 saturates term-frequency contribution and explicitly normalizes for document length, a
long document that repeats a query term many times (e.g. through keyword stuffing) will not
automatically outrank a shorter, genuinely on-topic document — something plain TF-IDF with raw term
counts is prone to.
    """,
    "procedure": [
        "Review the theoretical background, the ranking pipeline diagram, and the key terminology.",
        "Open the Simulation section from the sidebar.",
        "Choose a document source: the built-in corpus, your own pasted documents, or uploaded PDF / DOCX files.",
        "Enter a search query (a default query is pre-filled for the built-in corpus).",
        "Adjust the BM25 parameters k1 (term-frequency saturation) and b (length normalization).",
        "Watch the live simulation: the query is tokenized, matched against every document, scored, and ranked.",
        "Compare the TF-IDF and BM25 rankings, the score chart, and the rank correlation between the two methods.",
        "Click 'Record current trial' to log the query, parameters, and results.",
        "Repeat for 3-4 different queries and/or k1, b combinations.",
        "Complete the Quiz to test your conceptual understanding.",
        "Generate your PDF lab report, then your Certificate of Completion."
    ],
    "key_terms": {
        "Term Frequency (TF)": "Number of times a term occurs within a specific document.",
        "Document Frequency (DF)": "Number of documents in the collection that contain a given term.",
        "Inverse Document Frequency (IDF)": "A measure of how rare a term is across the collection; "
                                             "common terms get a lower weight, rare terms a higher one.",
        "avgdl": "The average document length (in tokens) across the entire corpus, used by BM25 for "
                 "length normalization.",
        "k1 (saturation parameter)": "BM25 parameter controlling how quickly repeated term occurrences "
                                      "saturate in their contribution to the score.",
        "b (length normalization parameter)": "BM25 parameter controlling how strongly a document's "
                                                "length (relative to avgdl) penalizes its score.",
        "Cosine Similarity": "The cosine of the angle between the query vector and a document vector; "
                              "used by the TF-IDF model to rank documents independent of vector magnitude.",
        "Rank Correlation": "A measure (Spearman's rho) of how similarly two ranking methods order the "
                             "same set of documents."
    }
}

BM25_APPLICATIONS = [
    ("Web & Enterprise Search", "BM25 is the default relevance-scoring function in Elasticsearch, "
     "OpenSearch, Apache Solr, and Apache Lucene — the engines behind most modern site search, "
     "intranet search, and log/analytics search deployments."),
    ("Retrieval-Augmented Generation (RAG)", "Large-language-model pipelines commonly use BM25 as a "
     "fast lexical first-stage retriever (often alongside a dense/embedding retriever) to fetch "
     "candidate passages that are then fed into the LLM as context."),
    ("Digital Libraries & Academic Search", "Scholarly search systems and digital library catalogues "
     "use BM25 to rank papers, articles, and citations against free-text queries."),
    ("E-Commerce Product Search", "Online marketplaces rank product listings against a shopper's "
     "search query using BM25-style scoring, often blended with popularity or sales signals."),
    ("Legal & Patent Retrieval", "Legal research platforms and patent search tools rely on BM25 to "
     "surface relevant case law, statutes, or prior art from large full-text document collections."),
    ("Open-Domain Question Answering", "QA systems use BM25 as the 'retriever' stage in a "
     "retrieve-then-read architecture, narrowing millions of documents down to a small candidate set "
     "before a reader model extracts the answer."),
]

REFERENCES_CONTENT = [
    {
        "citation": "S. E. Robertson and K. Sparck Jones, 'Relevance Weighting of Search Terms', "
                    "Journal of the American Society for Information Science, 1976.",
        "url": "https://www.semanticscholar.org/paper/Relevance-weighting-of-search-terms-Robertson-Jones/f6e3e57567e9803718623ec088cd7fea65cfbc9d"
    },
    {
        "citation": "S. E. Robertson and S. Walker, 'Some Simple Effective Approximations to the "
                    "2-Poisson Model for Probabilistic Weighted Retrieval', SIGIR, 1994.",
        "url": "https://www.staff.city.ac.uk/~sbrp622/papers/robertson_walker_sigir94.pdf"
    },
    {
        "citation": "C. D. Manning, P. Raghavan, and H. Schütze, 'Introduction to Information "
                    "Retrieval', Cambridge University Press, 2008 (Chapters 6 & 11: Scoring, Term "
                    "Weighting, and Probabilistic IR).",
        "url": "https://nlp.stanford.edu/IR-book/information-retrieval-book.html"
    },
]

COMPARISON_TABLE = [
    ("Underlying model", "Vector space / geometric similarity", "Probabilistic relevance model"),
    ("Term-frequency handling", "Unbounded (raw count); no saturation", "Saturates via parameter k1"),
    ("Document length handling", "Implicit, via vector normalization only", "Explicit normalization via parameter b"),
    ("Tunable parameters", "None (in the classic form used here)", "k1 and b"),
    ("Typical use", "Baseline / educational IR ranking", "Default ranking function in most modern search engines"),
]


# ======================================================================================
# 2. BUILT-IN CORPUS
# ======================================================================================

BUILTIN_CORPUS = [
    {"id": 1, "title": "Introduction to Machine Learning",
     "text": "Machine learning algorithms allow computers to learn patterns directly from data and "
             "improve their performance over time without being explicitly programmed for every task."},
    {"id": 2, "title": "Conference Summary Notes",
     "text": "This year's technology conference opened with a keynote on modern computing trends. "
             "Machine learning was mentioned throughout the sessions, and several speakers referenced "
             "machine learning as the future of the industry. Attendees discussed how machine learning "
             "is transforming business operations, how machine learning is being adopted across sectors, "
             "and how machine learning continues to attract investment. The closing panel praised machine "
             "learning once more before the networking reception, snacks, and evening social gathering for "
             "all conference attendees and sponsors in the main hall."},
    {"id": 3, "title": "Deep Learning Fundamentals",
     "text": "Deep learning uses multi layered neural networks to automatically extract hierarchical "
             "features from raw data, achieving strong results in image and speech recognition tasks."},
    {"id": 4, "title": "Natural Language Processing",
     "text": "Natural language processing combines linguistics and computation to help machines "
             "understand, interpret, and generate human language for tasks like translation and "
             "sentiment analysis."},
    {"id": 5, "title": "Computer Vision Applications",
     "text": "Computer vision enables systems to interpret and analyze visual information from images "
             "and video, supporting applications such as facial recognition and autonomous driving."},
    {"id": 6, "title": "Reinforcement Learning Basics",
     "text": "Reinforcement learning trains an agent to make sequential decisions by rewarding desirable "
             "actions and penalizing undesirable ones within a defined environment."},
    {"id": 7, "title": "Data Preprocessing Techniques",
     "text": "Before training a model, raw data must be cleaned, normalized, and transformed through "
             "preprocessing steps such as handling missing values and feature scaling."},
    {"id": 8, "title": "Neural Network Architectures",
     "text": "Neural networks consist of interconnected layers of nodes that process input signals "
             "through weighted connections and nonlinear activation functions. Common architectures "
             "include convolutional networks for spatial data, recurrent networks for sequential data, "
             "and transformer networks that rely on attention mechanisms. Choosing the right architecture "
             "depends heavily on the nature of the dataset and the learning task being solved by the model."},
    {"id": 9, "title": "Robotics and Automation",
     "text": "Robotics combines mechanical engineering and control systems to design machines capable of "
             "performing physical tasks, often guided by sensors and embedded software."},
    {"id": 10, "title": "Statistics and Probability Foundations",
     "text": "A solid foundation in statistics and probability theory underlies many machine learning "
             "algorithms, providing tools to quantify uncertainty and evaluate model performance."},
    {"id": 11, "title": "Cloud Computing Overview",
     "text": "Cloud computing provides on demand access to computing resources such as servers, storage, "
             "and networking over the internet, enabling scalable infrastructure for applications."},
    {"id": 12, "title": "History of Computing",
     "text": "The history of computing spans mechanical calculators, early electronic computers built "
             "with vacuum tubes, the invention of the transistor, and the eventual rise of personal "
             "computers and the internet. Each era introduced new capabilities that expanded what "
             "computation could achieve, setting the stage for later advances in software and data "
             "driven systems."},
    {"id": 13, "title": "Supervised and Unsupervised Learning",
     "text": "Supervised learning uses labeled examples to train predictive models, while unsupervised "
             "learning discovers hidden structure in unlabeled data through clustering and dimensionality "
             "reduction."},
    {"id": 14, "title": "Ethics in Artificial Intelligence",
     "text": "As machine learning systems increasingly influence decisions, researchers emphasize "
             "fairness, transparency, and accountability to reduce bias and ensure responsible artificial "
             "intelligence deployment."},
]

DEFAULT_QUERY = "machine learning algorithms"

SIMULATION_CONFIG = {
    "k1_min": 0.5,
    "k1_max": 3.0,
    "k1_default": 1.5,
    "k1_step": 0.1,
    "b_min": 0.0,
    "b_max": 1.0,
    "b_default": 0.75,
    "b_step": 0.05,
}

STOPWORDS = set("""
a an the and or but if while of in on for to with without within by from as is are was were be been
being this that these those it its into over under again further then once here there when where why
how all any both each few more most other some such no nor not only own same so than too very s t can
will just don should now which who whom
""".split())


QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "What does IDF (Inverse Document Frequency) measure in a ranking function?",
        "options": [
            "A) How frequently a term occurs within a single document",
            "B) How rare or common a term is across the entire document collection",
            "C) The total number of documents in the corpus",
            "D) The length of a document in tokens"
        ],
        "answer_index": 1,
        "explanation": "IDF down-weights terms that appear in many documents (common, less discriminating) "
                       "and up-weights terms that appear in few documents (rare, more discriminating)."
    },
    {
        "id": 2,
        "question": "In the classic vector-space TF-IDF model with raw term frequency, what happens to a "
                    "document's score as a query term is repeated many more times within it?",
        "options": [
            "A) The score keeps increasing roughly proportionally, with no saturation",
            "B) The score is capped after the term appears twice",
            "C) The score decreases due to an automatic length penalty",
            "D) The term is automatically removed as a stopword"
        ],
        "answer_index": 0,
        "explanation": "Raw term-frequency weighting has no saturation mechanism, so repeated occurrences "
                       "keep contributing roughly linearly to the score."
    },
    {
        "id": 3,
        "question": "What is the primary purpose of the BM25 parameter k1?",
        "options": [
            "A) It sets the number of documents to retrieve",
            "B) It controls how quickly term frequency's contribution to the score saturates",
            "C) It determines the size of the vocabulary",
            "D) It normalizes query length only"
        ],
        "answer_index": 1,
        "explanation": "k1 governs the saturation curve: larger k1 lets additional term occurrences keep "
                       "contributing longer before diminishing returns set in."
    },
    {
        "id": 4,
        "question": "What does the BM25 parameter b control?",
        "options": [
            "A) The weight given to inverse document frequency",
            "B) The degree of normalization for document length",
            "C) The number of query terms considered",
            "D) The minimum term frequency required for scoring"
        ],
        "answer_index": 1,
        "explanation": "b scales how strongly a document's length, relative to the average document "
                       "length (avgdl), penalizes its score."
    },
    {
        "id": 5,
        "question": "If b is set to 0 in BM25, what effect does this have?",
        "options": [
            "A) Document length normalization is completely disabled",
            "B) Term frequency saturation is disabled",
            "C) IDF is ignored entirely",
            "D) All documents receive an identical score"
        ],
        "answer_index": 0,
        "explanation": "With b = 0, the length-normalization term in the denominator reduces to 1, so "
                       "document length no longer affects the score."
    },
    {
        "id": 6,
        "question": "Why might a long document that repeats a query term many times score "
                    "disproportionately high under plain TF-IDF but not under BM25?",
        "options": [
            "A) TF-IDF ignores repeated terms while BM25 rewards them",
            "B) BM25 saturates term-frequency contribution and normalizes for document length, while raw "
            "TF-IDF has no such saturation",
            "C) TF-IDF cannot process long documents at all",
            "D) BM25 always assigns a score of zero to long documents"
        ],
        "answer_index": 1,
        "explanation": "BM25's k1 saturates repeated-term contributions and its b parameter normalizes for "
                       "document length, both of which counteract keyword-stuffing effects that raw "
                       "TF-IDF is susceptible to."
    },
    {
        "id": 7,
        "question": "Which of the following best distinguishes BM25 from the classic vector-space TF-IDF "
                    "model conceptually?",
        "options": [
            "A) BM25 is a probabilistic ranking function; TF-IDF's classic form is a geometric / "
            "vector-space similarity measure",
            "B) They are mathematically identical formulas",
            "C) TF-IDF uses IDF while BM25 does not use IDF at all",
            "D) BM25 only works for single-word queries"
        ],
        "answer_index": 0,
        "explanation": "TF-IDF with cosine similarity is rooted in the vector space model, while BM25 is "
                       "derived from a probabilistic model of relevance (the 2-Poisson model)."
    },
    {
        "id": 8,
        "question": "In BM25, what does avgdl (average document length) get compared against for each "
                    "document?",
        "options": [
            "A) The number of unique terms in the query",
            "B) That document's own length, to determine whether it is longer or shorter than typical "
            "documents in the collection",
            "C) The total vocabulary size of the corpus",
            "D) The number of documents containing the query term"
        ],
        "answer_index": 1,
        "explanation": "The ratio |D| / avgdl tells BM25 whether a document is longer or shorter than "
                       "average, which (scaled by b) shapes the length-normalization penalty."
    },
    {
        "id": 9,
        "question": "Cosine similarity, as used in the TF-IDF vector-space model, measures:",
        "options": [
            "A) The angle between the query vector and a document vector, independent of their magnitudes",
            "B) The raw dot product of two vectors including magnitude effects",
            "C) The Euclidean distance between two vectors",
            "D) The count of shared terms only"
        ],
        "answer_index": 0,
        "explanation": "Cosine similarity normalizes by both vectors' magnitudes, so it captures "
                       "orientation (similarity in direction/composition) rather than raw magnitude."
    },
    {
        "id": 10,
        "question": "Which scenario would you expect to benefit most from tuning BM25's b parameter "
                    "upward (closer to 1)?",
        "options": [
            "A) A corpus where all documents have exactly the same length",
            "B) A corpus with highly varying document lengths, where longer documents might otherwise "
            "dominate purely by repeating terms",
            "C) A corpus with only one document",
            "D) A corpus with no query terms present in any document"
        ],
        "answer_index": 1,
        "explanation": "Length normalization matters most when document lengths vary widely; b closer to 1 "
                       "more strongly discounts long documents relative to the corpus average."
    }
]


# ======================================================================================
# 3. RANKING ENGINE (TF-IDF cosine similarity + BM25)
# ======================================================================================

def tokenize(text: str) -> list:
    """Lowercases, extracts alphabetic tokens, and removes a small built-in stopword list."""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def spearman_rank_correlation(order_a: list, order_b: list) -> float:
    """
    Computes Spearman's rank correlation between two rankings of the same items
    (given as lists of doc ids in ranked order), without external dependencies.
    """
    ids = order_a
    n = len(ids)
    if n < 2:
        return 1.0
    rank_a = {doc_id: i for i, doc_id in enumerate(order_a)}
    rank_b = {doc_id: i for i, doc_id in enumerate(order_b)}
    d2_sum = sum((rank_a[doc_id] - rank_b[doc_id]) ** 2 for doc_id in ids)
    return round(1 - (6 * d2_sum) / (n * (n ** 2 - 1)), 4)


def rank_documents(documents: list, query: str, k1: float, b: float) -> dict:
    """
    Scores every document in `documents` against `query` using both TF-IDF cosine
    similarity and BM25, returning per-document results plus supporting stats.
    """
    doc_tokens = [tokenize(d["text"]) for d in documents]
    doc_counts = [collections.Counter(toks) for toks in doc_tokens]
    doc_lengths = [len(toks) for toks in doc_tokens]
    n_docs = len(documents)
    avgdl = (sum(doc_lengths) / n_docs) if n_docs else 0.0

    df = collections.Counter()
    for counts in doc_counts:
        for term in counts:
            df[term] += 1

    def idf_tfidf(term: str) -> float:
        dfi = df.get(term, 0)
        if dfi == 0:
            return 0.0
        return math.log(n_docs / dfi) + 1.0

    def idf_bm25(term: str) -> float:
        dfi = df.get(term, 0)
        return math.log(1.0 + (n_docs - dfi + 0.5) / (dfi + 0.5))

    query_tokens = tokenize(query)
    query_counts = collections.Counter(query_tokens)
    unique_terms = list(query_counts.keys())

    q_weights = {t: query_counts[t] * idf_tfidf(t) for t in unique_terms}
    q_norm = math.sqrt(sum(w * w for w in q_weights.values()))

    results = []
    for i, doc in enumerate(documents):
        counts = doc_counts[i]
        dl = doc_lengths[i]

        d_norm_sq = 0.0
        dot = 0.0
        for term, tf in counts.items():
            w = tf * idf_tfidf(term)
            d_norm_sq += w * w
            if term in q_weights:
                dot += w * q_weights[term]
        d_norm = math.sqrt(d_norm_sq)
        tfidf_score = (dot / (d_norm * q_norm)) if d_norm > 0 and q_norm > 0 else 0.0

        bm25_score = 0.0
        term_contributions = {}
        for term in unique_terms:
            f = counts.get(term, 0)
            if f == 0:
                continue
            idf = idf_bm25(term)
            denom = f + k1 * (1 - b + b * (dl / avgdl if avgdl > 0 else 0.0))
            contrib = (idf * f * (k1 + 1)) / denom if denom > 0 else 0.0
            bm25_score += contrib
            term_contributions[term] = round(contrib, 4)

        results.append({
            "id": doc["id"],
            "title": doc["title"],
            "length": dl,
            "tfidf_score": round(tfidf_score, 4),
            "bm25_score": round(bm25_score, 4),
            "term_contributions": term_contributions
        })

    return {
        "results": results,
        "unique_query_terms": unique_terms,
        "avgdl": round(avgdl, 2),
        "num_documents": n_docs
    }


def parse_custom_documents(raw_text: str) -> list:
    """Splits a pasted block of text into documents, one per blank-line-separated paragraph."""
    chunks = [c.strip() for c in re.split(r"\n\s*\n", raw_text) if c.strip()]
    return [{"id": i + 1, "title": f"Custom Document {i + 1}", "text": chunk} for i, chunk in enumerate(chunks)]


def extract_text_from_upload(uploaded_file) -> str:
    """Extracts plain text from an uploaded PDF or DOCX file object. Returns '' on failure."""
    name = (uploaded_file.name or "").lower()
    try:
        if name.endswith(".pdf"):
            if not PYPDF_AVAILABLE:
                return ""
            reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
            return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        elif name.endswith(".docx"):
            if not DOCX_AVAILABLE:
                return ""
            doc = DocxDocument(io.BytesIO(uploaded_file.getvalue()))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
    except Exception:
        return ""
    return ""


def build_embedding_projection(documents: list, query: str, retrieved_ids: set) -> dict:
    """
    Builds a lightweight 2D 'embedding' of every document and the query by projecting their
    TF-IDF vectors down using SVD (a PCA-like technique), purely for visualization purposes.
    Reuses the same IDF weighting as the TF-IDF ranking engine above.
    """
    doc_tokens = [tokenize(d["text"]) for d in documents]
    doc_counts = [collections.Counter(toks) for toks in doc_tokens]
    n_docs = len(documents)

    df = collections.Counter()
    for counts in doc_counts:
        for term in counts:
            df[term] += 1

    def idf_tfidf(term: str) -> float:
        dfi = df.get(term, 0)
        if dfi == 0:
            return 0.0
        return math.log(n_docs / dfi) + 1.0

    query_tokens = tokenize(query)
    query_counts = collections.Counter(query_tokens)

    vocab = sorted(df.keys())
    if not vocab:
        return {"points": []}
    vocab_index = {term: i for i, term in enumerate(vocab)}

    matrix = np.zeros((n_docs + 1, len(vocab)))
    for i, counts in enumerate(doc_counts):
        for term, tf in counts.items():
            matrix[i, vocab_index[term]] = tf * idf_tfidf(term)
    for term, tf in query_counts.items():
        if term in vocab_index:
            matrix[n_docs, vocab_index[term]] = tf * idf_tfidf(term)

    # Center the matrix and project onto its top-2 singular vectors (PCA-like 2D embedding).
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    try:
        u, s, _vt = np.linalg.svd(centered, full_matrices=False)
        coords_2d = u[:, :2] * s[:2]
    except np.linalg.LinAlgError:
        coords_2d = np.zeros((n_docs + 1, 2))

    if coords_2d.shape[1] < 2:
        coords_2d = np.pad(coords_2d, ((0, 0), (0, 2 - coords_2d.shape[1])))

    points = []
    for i, doc in enumerate(documents):
        points.append({
            "id": doc["id"],
            "title": doc["title"],
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1]),
            "retrieved": doc["id"] in retrieved_ids
        })
    points.append({
        "id": "query", "title": "QUERY", "x": float(coords_2d[n_docs, 0]),
        "y": float(coords_2d[n_docs, 1]), "retrieved": False
    })
    return {"points": points}


# ======================================================================================
# 4. LAB REPORT PDF EXPORTER
# ======================================================================================

class LabReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Virtual Laboratory Report", align="C")


def generate_pdf_report(student_name: str, student_id: str, date_str: str,
                        trials_df: pd.DataFrame, quiz_score: int, quiz_total: int,
                        student_notes: str) -> bytes:
    """Compiles experiment trial records into a formatted PDF report document."""
    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, EXPERIMENT_CONFIG["title"], align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 22, 190, 22, "FD")

    pdf.set_xy(14, 24)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Student Name:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, student_name or "N/A", 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Student ID / Roll:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(50, 5, student_id or "N/A", 1)

    pdf.set_xy(14, 32)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Experiment Date:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, date_str or datetime.now().strftime("%Y-%m-%d"), 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Quiz Evaluation:", 0)
    pdf.set_font("Helvetica", "B", 9)
    if quiz_score >= max(1, quiz_total // 2):
        pdf.set_text_color(16, 185, 129)
    else:
        pdf.set_text_color(239, 68, 68)
    pdf.cell(50, 5, f"{quiz_score} / {quiz_total} ({int((quiz_score/quiz_total)*100 if quiz_total else 0)}%)", 1)

    pdf.ln(12)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "1. Learning Objectives", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    for obj in EXPERIMENT_CONFIG["objectives"]:
        clean_obj = str(obj).replace("$", "").replace("\\", "")
        pdf.cell(5, 5, "-", 0)
        pdf.cell(0, 5, f" {clean_obj}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "2. Recorded Experimental Trials & Data", new_x="LMARGIN", new_y="NEXT")

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No simulation trials recorded during this session.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_fill_color(37, 99, 235)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 7)

        cols = list(trials_df.columns)
        num_cols = len(cols)
        col_w = max(15, int(190 / max(1, num_cols)))

        for c in cols:
            pdf.cell(col_w, 6, str(c)[:16], 1, 0, "C", True)
        pdf.ln()

        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 7)
        fill = False

        for _, row in trials_df.iterrows():
            for c in cols:
                val = row[c]
                val_str = f"{val:.3f}" if isinstance(val, float) else str(val)
                pdf.cell(col_w, 5, val_str[:16], 1, 0, "C", fill)
            pdf.ln()
            fill = not fill
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "3. Observations & Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    notes_text = student_notes.strip() if student_notes.strip() else (
        "The BM25 and TF-IDF rankings were compared across several queries and parameter settings, "
        "with BM25 showing greater robustness to document length and repeated-term keyword stuffing."
    )
    pdf.multi_cell(0, 5, notes_text)
    pdf.ln(8)

    pdf.set_draw_color(180, 180, 180)
    pdf.line(130, pdf.get_y() + 15, 190, pdf.get_y() + 15)
    pdf.set_xy(130, pdf.get_y() + 17)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(60, 4, "Instructor / Student Signature", align="C")

    return bytes(pdf.output())


def generate_certificate_pdf(student_name: str, quiz_score: int, quiz_total: int, date_str: str) -> bytes:
    """Builds a landscape A4 'Certificate of Completion' PDF for a student who finished the lab."""
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    page_w, page_h = 297, 210
    perc = int((quiz_score / quiz_total) * 100) if quiz_total else 0

    # Decorative outer and inner border.
    pdf.set_draw_color(30, 58, 138)
    pdf.set_line_width(1.2)
    pdf.rect(8, 8, page_w - 16, page_h - 16)
    pdf.set_draw_color(37, 99, 235)
    pdf.set_line_width(0.4)
    pdf.rect(12, 12, page_w - 24, page_h - 24)

    pdf.set_text_color(30, 58, 138)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_xy(0, 24)
    pdf.cell(page_w, 8, "KGIRS VIRTUAL LABORATORY", align="C")

    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 30)
    pdf.set_xy(0, 40)
    pdf.cell(page_w, 16, "Certificate of Completion", align="C")

    pdf.set_draw_color(203, 213, 225)
    pdf.set_line_width(0.3)
    pdf.line(page_w / 2 - 40, 60, page_w / 2 + 40, 60)

    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(71, 85, 105)
    pdf.set_xy(0, 72)
    pdf.cell(page_w, 8, "This is to certify that", align="C")

    clean_name = (student_name or "Student").replace("$", "").replace("\\", "")
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(30, 58, 138)
    pdf.set_xy(0, 84)
    pdf.cell(page_w, 14, clean_name, align="C")
    pdf.set_draw_color(30, 58, 138)
    pdf.set_line_width(0.4)
    pdf.line(page_w / 2 - 55, 100, page_w / 2 + 55, 100)

    pdf.set_font("Helvetica", "", 12.5)
    pdf.set_text_color(51, 65, 85)
    pdf.set_xy(35, 108)
    pdf.multi_cell(
        page_w - 70, 7,
        f"has successfully completed the Virtual Laboratory experiment on "
        f"\"{EXPERIMENT_CONFIG['title']}\", performing the interactive simulation and completing the "
        f"concept assessment quiz with a final score of {quiz_score} / {quiz_total} ({perc}%).",
        align="C"
    )

    pdf.set_font("Helvetica", "B", 12)
    if perc >= 50:
        pdf.set_text_color(16, 185, 129)
    else:
        pdf.set_text_color(239, 68, 68)
    pdf.set_xy(0, 138)
    pdf.cell(page_w, 8, f"Quiz Score: {quiz_score} / {quiz_total}  ({perc}%)", align="C")

    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.set_xy(0, 150)
    pdf.cell(page_w, 6, f"Date of Completion: {date_str}", align="C")

    pdf.set_draw_color(180, 180, 180)
    pdf.set_line_width(0.3)
    pdf.line(45, 178, 110, 178)
    pdf.set_xy(45, 180)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(65, 5, "Student Signature", align="C")

    pdf.line(page_w - 110, 178, page_w - 45, 178)
    pdf.set_xy(page_w - 110, 180)
    pdf.cell(65, 5, "Instructor / Lab Coordinator", align="C")

    return bytes(pdf.output())


def is_lab_completed() -> bool:
    """A student has completed the lab once they've recorded at least one simulation trial and
    submitted the quiz."""
    has_trial = bool(st.session_state.get("trials"))
    has_quiz = bool(st.session_state.get("quiz_submitted", False))
    return has_trial and has_quiz


# ======================================================================================
# 5. UI SYSTEM: DESIGN TOKENS, GLOBAL STYLES, LAYOUT PRIMITIVES
# ======================================================================================

NAV_ITEMS = [
    ("purpose", "Purpose", ":material/flag:"),
    ("theory", "Theory", ":material/menu_book:"),
    ("simulation", "Simulation", ":material/science:"),
    ("quiz", "Quiz", ":material/quiz:"),
    ("report", "Report Generation", ":material/description:"),
    ("certificate", "Certificate", ":material/workspace_premium:"),
    ("references", "References", ":material/library_books:"),
]
NAV_KEYS = [k for k, _, _ in NAV_ITEMS]
NAV_LABELS = {k: f"{icon}&nbsp;&nbsp;{label}" for k, label, icon in NAV_ITEMS}

# Chart colours: validated categorical pair (blue / orange) + neutral, and text inks.
SERIES = {"bm25": "#2a78d6", "tfidf": "#eb6834", "neutral": "#b8c0cc"}
INK = {"primary": "#0f172a", "secondary": "#475569", "muted": "#64748b",
       "grid": "#eef2f7", "border": "#e2e8f0"}
FONT_STACK = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

# Widget values that should survive navigating between sections.
PERSIST_KEYS = ["sim_source", "q_builtin", "q_paste", "q_upload", "paste_text", "k1", "b",
                "rep_name", "rep_id", "rep_date", "rep_notes", "cert_name"]

SOURCE_OPTIONS = ["Built-in corpus", "Paste documents", "Upload PDF / DOCX"]

ICON_CHECK = ('<svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" '
              'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">'
              '<path d="M3.5 8.5l3 3 6-7"/></svg>')
ICON_LOCK = ('<svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" '
             'stroke-width="1.8" stroke-linecap="round"><rect x="3.5" y="7" width="9" height="6.5" rx="1.5"/>'
             '<path d="M5.5 7V5a2.5 2.5 0 015 0v2"/></svg>')
ICON_ARROW = ('<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" '
              'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
              '<path d="M6 3.5h6.5V10M12.5 3.5L4 12"/></svg>')

APP_CSS = """
<style>
:root {
  --vl-primary: #2563eb; --vl-primary-600: #1d4ed8; --vl-primary-50: #eff6ff; --vl-primary-100: #dbeafe;
  --vl-ink: #0f172a; --vl-ink-2: #334155; --vl-muted: #64748b; --vl-border: #e2e8f0;
  --vl-surface: #f8fafc; --vl-radius: 12px;
  --vl-good: #15803d; --vl-good-50: #f0fdf4; --vl-warn: #b45309; --vl-warn-50: #fffbeb;
}

/* ---------- App chrome ---------- */
[data-testid="stToolbarActions"], [data-testid="stAppDeployButton"], [data-testid="stMainMenu"],
[data-testid="stStatusWidget"], [data-testid="stDecoration"], [data-testid="stHeaderActionElements"] {
  display: none !important;
}
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stMainBlockContainer"] { max-width: 1180px; padding: 2.4rem 2.5rem 4rem 2.5rem; }
@media (max-width: 640px) { [data-testid="stMainBlockContainer"] { padding: 2.2rem 1rem 3rem 1rem; } }

/* No dimming / fading of the previous page while a rerun is in flight (prevents flicker). */
[data-stale="true"], .stale-element { opacity: 1 !important; transition: none !important; filter: none !important; }
[data-testid="stElementContainer"] { transition: none !important; }

/* ---------- Sidebar: fixed, non-scrolling, flex column ---------- */
section[data-testid="stSidebar"] { border-right: 1px solid var(--vl-border); }
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  overflow: hidden !important; display: flex; flex-direction: column; height: 100vh; height: 100dvh;
  scrollbar-width: none;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar { display: none; }
[data-testid="stSidebarHeader"] { height: 2.6rem; min-height: 2.6rem; padding: 0.6rem 0.9rem 0 1rem; margin: 0; }
[data-testid="stSidebarUserContent"] {
  padding: 0 1rem 1rem 1rem !important; flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column;
}
[data-testid="stSidebarUserContent"] > div { flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] { flex: 1 1 auto; min-height: 0; gap: 0.35rem; }
[data-testid="stSidebar"] *:has(> .st-key-sb_progress) { margin-top: auto; }
.st-key-sb_progress { padding-bottom: 0.9rem; }
[data-testid="stSidebar"] [data-testid="stElementContainer"] { width: 100% !important; }

.vl-brand { display: flex; align-items: center; gap: 0.7rem; padding: 0 0.25rem 1rem 0.25rem;
            margin-bottom: 0.6rem; border-bottom: 1px solid var(--vl-border); }
.vl-logo { width: 38px; height: 38px; border-radius: 10px; flex: 0 0 38px; display: grid; place-items: center;
           background: linear-gradient(135deg, #2563eb 0%, #1e3a8a 100%); color: #fff;
           box-shadow: 0 2px 6px rgba(37, 99, 235, .25); }
.vl-brand-title { font-weight: 700; font-size: 0.98rem; color: var(--vl-ink); letter-spacing: -0.01em; line-height: 1.2; }
.vl-brand-sub { font-size: 0.78rem; color: var(--vl-muted); margin-top: 2px; line-height: 1.25; }
.vl-nav-label { font-size: 0.7rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
                color: #94a3b8; padding: 0.35rem 0.5rem 0.1rem; }

/* Navigation (radio restyled as a nav list) */
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stRadio"]) { width: 100% !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] { width: 100%; }
[data-testid="stSidebar"] [data-testid="stRadio"] > [data-testid="stWidgetLabel"] { display: none; }
[data-testid="stSidebar"] [data-testid="stRadioGroup"] { gap: 2px; width: 100%; }
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div { width: 100%; }
[data-testid="stSidebar"] [data-testid="stRadioOption"] {
  width: 100%; margin: 0; padding: 0.5rem 0.7rem; border-radius: 8px; cursor: pointer;
  transition: background-color .15s ease;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child { display: none; }
[data-testid="stSidebar"] [data-testid="stRadioOption"] p {
  font-size: 0.9rem; font-weight: 500; color: var(--vl-ink-2); display: flex; align-items: center;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] [data-testid="stIconMaterial"] { color: #94a3b8; font-size: 1.15rem; }
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover { background: #eef2f7; }
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] { background: var(--vl-primary-50); }
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p { color: var(--vl-primary-600); font-weight: 600; }
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] [data-testid="stIconMaterial"] { color: var(--vl-primary); }

/* Progress card */
.vl-progress { border: 1px solid var(--vl-border); background: #fff; border-radius: var(--vl-radius); padding: 0.85rem 0.9rem; }
.vl-progress-head { display: flex; justify-content: space-between; align-items: baseline; font-size: 0.8rem;
                    font-weight: 600; color: var(--vl-ink); }
.vl-progress-head span:last-child { color: var(--vl-muted); font-weight: 500; font-variant-numeric: tabular-nums; }
.vl-bar { height: 6px; border-radius: 999px; background: #eef2f7; overflow: hidden; margin: 0.55rem 0 0.7rem; }
.vl-bar i { display: block; height: 100%; border-radius: 999px; background: var(--vl-primary); transition: width .5s ease; }
.vl-steps { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.45rem; }
.vl-steps li { display: flex; align-items: center; gap: 0.5rem; font-size: 0.8rem; color: var(--vl-muted); margin: 0; }
.vl-steps li .dot { width: 18px; height: 18px; flex: 0 0 18px; border-radius: 50%; display: grid; place-items: center;
                    border: 1.5px solid #cbd5e1; color: #94a3b8; }
.vl-steps li.done { color: var(--vl-ink-2); }
.vl-steps li.done .dot { background: var(--vl-good); border-color: var(--vl-good); color: #fff; }
.vl-steps li em { margin-left: auto; font-style: normal; font-size: 0.74rem; color: var(--vl-muted); font-variant-numeric: tabular-nums; }

/* Short / portrait viewports: compress instead of scrolling */
@media (max-height: 780px) {
  [data-testid="stSidebar"] [data-testid="stRadioOption"] { padding: 0.38rem 0.7rem; }
  .vl-brand { padding-bottom: 0.7rem; }
  .vl-progress { padding: 0.7rem 0.8rem; }
}
@media (max-height: 660px) {
  .vl-nav-label, .vl-brand-sub, .vl-steps li em { display: none; }
  [data-testid="stSidebar"] [data-testid="stRadioOption"] { padding: 0.3rem 0.7rem; }
  .vl-steps { gap: 0.3rem; }
}
@media (max-height: 560px) { .vl-steps { display: none; } .vl-bar { margin-bottom: 0; } }

/* ---------- Page header ---------- */
@keyframes vlIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
.vl-hero { padding: 0 0 1.2rem; margin-bottom: 1.1rem; border-bottom: 1px solid var(--vl-border); animation: vlIn .35s ease both; }
.vl-eyebrow { font-size: 0.74rem; letter-spacing: 0.09em; text-transform: uppercase; font-weight: 600; color: var(--vl-primary); }
.vl-hero h1 { font-size: 2rem; font-weight: 700; letter-spacing: -0.025em; line-height: 1.15; margin: 0.35rem 0 0.4rem;
              padding: 0; color: var(--vl-ink); }
.vl-hero p { color: var(--vl-muted); font-size: 1rem; line-height: 1.55; margin: 0; max-width: 760px; }

/* ---------- Cards & typography ---------- */
[class*="st-key-card"] {
  background: #fff; border: 1px solid var(--vl-border); border-radius: var(--vl-radius);
  padding: 1.25rem 1.4rem 1.3rem; box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
}
.vl-card-title { display: flex; flex-direction: column; gap: 2px; margin-bottom: 0.15rem; }
.vl-card-title h3 { font-size: 1.05rem; font-weight: 650; color: var(--vl-ink); margin: 0; padding: 0; letter-spacing: -0.01em; }
.vl-card-title span { font-size: 0.86rem; color: var(--vl-muted); }
.vl-h2 { font-size: 1.12rem; font-weight: 650; color: var(--vl-ink); margin: 1.1rem 0 0.1rem; letter-spacing: -0.01em; }
.vl-lead { color: var(--vl-muted); font-size: 0.92rem; margin: 0 0 0.2rem; }

.stMarkdown h3:not(.vl-card-title h3) { font-size: 1.12rem !important; font-weight: 650 !important; letter-spacing: -0.01em; padding-top: 0.9rem !important; }
.stMarkdown blockquote { border-left: 3px solid var(--vl-primary); background: var(--vl-surface); padding: 0.6rem 1rem;
                         border-radius: 0 8px 8px 0; margin: 0.6rem 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.86rem; }
.stMarkdown blockquote p { font-size: 0.86rem; margin: 0; }

/* Aim */
.vl-aim { display: flex; gap: 1rem; align-items: flex-start; }
.vl-aim-mark { width: 40px; height: 40px; flex: 0 0 40px; border-radius: 10px; background: var(--vl-primary-50);
               color: var(--vl-primary); display: grid; place-items: center; }
.vl-aim .k { font-size: 0.74rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--vl-muted); }
.vl-aim p { font-size: 1.08rem; line-height: 1.6; color: var(--vl-ink); margin: 0.2rem 0 0; font-weight: 500; }

/* Grids */
.vl-grid { display: grid; gap: 0.8rem; }
.vl-grid.c2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.vl-grid.c3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.vl-grid.c4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
@media (max-width: 1100px) { .vl-grid.c4 { grid-template-columns: repeat(2, minmax(0, 1fr)); } .vl-grid.c3 { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 700px) { .vl-grid.c2, .vl-grid.c3, .vl-grid.c4 { grid-template-columns: 1fr; } }

.vl-tile { border: 1px solid var(--vl-border); border-radius: var(--vl-radius); background: #fff; padding: 1rem 1.1rem; }
.vl-tile .num { font-size: 0.78rem; font-weight: 700; color: var(--vl-primary); font-variant-numeric: tabular-nums; }
.vl-tile h4 { font-size: 0.95rem; font-weight: 650; color: var(--vl-ink); margin: 0.3rem 0 0.25rem; padding: 0; }
.vl-tile p { font-size: 0.87rem; color: var(--vl-ink-2); line-height: 1.55; margin: 0; }

.vl-stat { border: 1px solid var(--vl-border); border-radius: var(--vl-radius); background: #fff; padding: 0.85rem 1rem; min-width: 0; }
.vl-stat .k { font-size: 0.78rem; color: var(--vl-muted); font-weight: 500; }
.vl-stat .v { font-size: 1.3rem; font-weight: 700; color: var(--vl-ink); margin-top: 0.2rem; letter-spacing: -0.015em;
              line-height: 1.25; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.vl-stat .v.sm { font-size: 1rem; font-weight: 650; }
.vl-stat .s { font-size: 0.76rem; color: var(--vl-muted); margin-top: 0.2rem; }
.vl-stat abbr { text-decoration: underline dotted; text-underline-offset: 3px; cursor: help; }

/* Flow of steps */
.vl-flow { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 0.6rem; counter-reset: flow; }
.vl-flow div { border: 1px solid var(--vl-border); border-radius: 10px; padding: 0.75rem 0.85rem; background: var(--vl-surface); }
.vl-flow b { display: block; font-size: 0.88rem; color: var(--vl-ink); font-weight: 650; }
.vl-flow b::before { counter-increment: flow; content: counter(flow); display: inline-grid; place-items: center; width: 20px; height: 20px;
                     margin-right: 0.45rem; border-radius: 50%; background: var(--vl-primary); color: #fff; font-size: 0.72rem; }
.vl-flow span { display: block; font-size: 0.8rem; color: var(--vl-muted); margin-top: 0.3rem; line-height: 1.45; }
@media (max-width: 1000px) { .vl-flow { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

/* Tables */
.vl-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.88rem; border: 1px solid var(--vl-border);
            border-radius: 10px; overflow: hidden; }
.vl-table th { text-align: left; font-weight: 600; font-size: 0.78rem; letter-spacing: .03em; color: var(--vl-muted);
               background: var(--vl-surface); padding: 0.6rem 0.85rem; border-bottom: 1px solid var(--vl-border); }
.vl-table td { padding: 0.6rem 0.85rem; border-bottom: 1px solid var(--vl-border); color: var(--vl-ink-2); vertical-align: top; }
.vl-table tr:last-child td { border-bottom: 0; }
.vl-table td:first-child { font-weight: 600; color: var(--vl-ink); }

/* Ordered procedure */
.vl-steps-list { list-style: none; counter-reset: s; margin: 0.3rem 0 0; padding: 0; display: grid; gap: 0.45rem; }
.vl-steps-list li { counter-increment: s; display: flex; gap: 0.75rem; align-items: flex-start; font-size: 0.92rem; color: var(--vl-ink-2); margin: 0; line-height: 1.5; }
.vl-steps-list li::before { content: counter(s); flex: 0 0 24px; height: 24px; border-radius: 50%; display: grid; place-items: center;
                            background: var(--vl-primary-50); color: var(--vl-primary-600); font-size: 0.75rem; font-weight: 700; }

/* References */
.vl-ref { display: flex; gap: 1rem; align-items: flex-start; border: 1px solid var(--vl-border); border-radius: var(--vl-radius);
          padding: 1rem 1.15rem; background: #fff; margin-bottom: 0.7rem; }
.vl-ref .idx { flex: 0 0 30px; height: 30px; border-radius: 8px; display: grid; place-items: center; background: var(--vl-surface);
               font-weight: 700; font-size: 0.8rem; color: var(--vl-ink-2); }
.vl-ref p { margin: 0; font-size: 0.93rem; color: var(--vl-ink); line-height: 1.55; }
.vl-ref a { display: inline-flex; gap: 0.3rem; align-items: center; margin-top: 0.45rem; font-size: 0.84rem; font-weight: 600;
            color: var(--vl-primary) !important; text-decoration: none; }
.vl-ref a:hover { text-decoration: underline; }

/* Requirement checklist (certificate) */
.vl-req { display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 0.9rem; border: 1px solid var(--vl-border);
          border-radius: 10px; margin-bottom: 0.5rem; background: #fff; }
.vl-req .ic { width: 26px; height: 26px; flex: 0 0 26px; border-radius: 50%; display: grid; place-items: center; }
.vl-req.ok .ic { background: var(--vl-good); color: #fff; }
.vl-req.no .ic { background: #f1f5f9; color: #64748b; }
.vl-req b { font-size: 0.92rem; color: var(--vl-ink); font-weight: 600; display: block; }
.vl-req span { font-size: 0.82rem; color: var(--vl-muted); }
.vl-req .tag { margin-left: auto; font-size: 0.74rem; font-weight: 600; padding: 0.2rem 0.55rem; border-radius: 999px; }
.vl-req.ok .tag { background: var(--vl-good-50); color: var(--vl-good); }
.vl-req.no .tag { background: var(--vl-warn-50); color: var(--vl-warn); }

/* Certificate preview */
.vl-cert { border: 1px solid #c7d2fe; outline: 4px solid #eef2ff; border-radius: 10px; padding: 2.4rem 2rem; text-align: center;
           background: radial-gradient(ellipse at top, #f8fbff 0%, #ffffff 60%); animation: vlIn .4s ease both; }
.vl-cert .org { font-size: 0.74rem; letter-spacing: .18em; font-weight: 700; color: #1e3a8a; }
.vl-cert h2 { font-size: 2rem; font-weight: 700; color: var(--vl-ink); margin: 0.5rem 0 0.2rem; padding: 0; letter-spacing: -0.02em; }
.vl-cert .rule { width: 72px; height: 2px; background: #cbd5e1; margin: 0.9rem auto; }
.vl-cert .pre { color: var(--vl-muted); font-size: 0.95rem; }
.vl-cert .name { font-size: 1.75rem; font-weight: 700; color: #1e3a8a; margin: 0.45rem 0 0; }
.vl-cert .line { width: 220px; height: 1.5px; background: #1e3a8a; margin: 0.4rem auto 1rem; opacity: .7; }
.vl-cert .body { max-width: 640px; margin: 0 auto; color: var(--vl-ink-2); line-height: 1.65; font-size: 0.97rem; }
.vl-cert .score { display: inline-block; margin-top: 1rem; padding: 0.35rem 0.9rem; border-radius: 999px; background: var(--vl-good-50);
                  color: var(--vl-good); font-weight: 700; font-size: 0.9rem; }
.vl-cert .date { margin-top: 0.8rem; font-size: 0.85rem; color: var(--vl-muted); }

/* Quiz */
[class*="st-key-card_q"] { padding: 1rem 1.2rem 0.6rem; }
.vl-q { font-size: 0.78rem; font-weight: 700; color: var(--vl-primary); letter-spacing: .04em; }
.vl-qt { font-size: 1rem; font-weight: 600; color: var(--vl-ink); margin: 0.2rem 0 0.1rem; line-height: 1.5; }
.vl-score { display: flex; align-items: center; gap: 1.2rem; }
.vl-score .big { font-size: 2.3rem; font-weight: 800; letter-spacing: -0.03em; color: var(--vl-ink); font-variant-numeric: tabular-nums; }
.vl-score .meta { flex: 1; }
.vl-score .meta b { font-size: 0.95rem; color: var(--vl-ink); }
.vl-score .meta .vl-bar { margin: 0.5rem 0 0.2rem; height: 8px; }
.vl-fb { border: 1px solid var(--vl-border); border-left-width: 3px; border-radius: 8px; padding: 0.7rem 0.9rem; margin-bottom: 0.5rem; background: #fff; }
.vl-fb.ok { border-left-color: var(--vl-good); }
.vl-fb.no { border-left-color: #dc2626; }
.vl-fb .h { font-size: 0.86rem; font-weight: 650; color: var(--vl-ink); display: flex; gap: 0.5rem; align-items: center; }
.vl-fb .h i { font-style: normal; font-size: 0.72rem; font-weight: 700; padding: 0.1rem 0.45rem; border-radius: 999px; }
.vl-fb.ok .h i { background: var(--vl-good-50); color: var(--vl-good); }
.vl-fb.no .h i { background: #fef2f2; color: #b91c1c; }
.vl-fb p { margin: 0.3rem 0 0; font-size: 0.85rem; color: var(--vl-ink-2); line-height: 1.5; }

/* Markdown-rendered components: neutralise default markdown spacing */
.vl-hero h1, .vl-card-title h3, .vl-tile h4, .vl-cert h2 { padding: 0 !important; }
.vl-hero h1 { font-size: 2rem !important; font-weight: 700 !important; margin: 0.35rem 0 0.4rem !important; line-height: 1.15 !important; }
.vl-card-title h3 { font-size: 1.05rem !important; font-weight: 650 !important; margin: 0 !important; line-height: 1.35 !important; color: var(--vl-ink) !important; }
.vl-card-title { margin-bottom: 0.35rem !important; }
.vl-tile h4 { font-size: 0.95rem !important; font-weight: 650 !important; margin: 0.3rem 0 0.25rem !important; }
.vl-cert h2 { font-size: 2rem !important; font-weight: 700 !important; margin: 0.5rem 0 0.2rem !important; }
.vl-hero p, .vl-tile p, .vl-aim p, .vl-ref p, .vl-fb p, .vl-lead { margin-bottom: 0 !important; }
.vl-hero p { margin-top: 0 !important; font-size: 1rem !important; }
.vl-aim p { margin-top: 0.2rem !important; font-size: 1.08rem !important; }
.vl-tile p { font-size: 0.87rem !important; }
.vl-ref p { font-size: 0.93rem !important; }
.vl-fb p { margin-top: 0.3rem !important; font-size: 0.85rem !important; }
.vl-lead { font-size: 0.92rem !important; margin-top: 0 !important; }
.vl-steps li, .vl-steps-list li { margin: 0 !important; }
.vl-steps-list li { font-size: 0.92rem !important; }
.vl-steps li { font-size: 0.8rem !important; }
ul.vl-steps, ol.vl-steps-list { padding-left: 0 !important; margin-left: 0 !important; }

/* Widgets */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button { font-weight: 600; border-radius: 8px; transition: all .15s ease; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 0.25rem; border-bottom: 1px solid var(--vl-border); }
[data-testid="stTabs"] [data-baseweb="tab"] { padding: 0.55rem 0.8rem; font-weight: 500; }
[data-testid="stTabs"] [data-baseweb="tab"] p { font-size: 0.88rem; }
[data-testid="stExpander"] details { border-radius: 10px; }
.st-key-vl_scroll { position: absolute !important; height: 0 !important; overflow: hidden !important; }
</style>
"""


def inject_css():
    st.html(APP_CSS)


def ui(markup: str):
    """Renders trusted HTML (dynamic values are escaped with esc()). st.markdown keeps inline SVG icons."""
    st.markdown(markup, unsafe_allow_html=True)


def esc(text) -> str:
    return _html.escape(str(text), quote=True)


def page_header(title: str, subtitle: str):
    ui(
        f'<div class="vl-hero"><div class="vl-eyebrow">KGIRS Virtual Lab &middot; Experiment 5</div>'
        f'<h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>'
    )


def card(key: str):
    """A white, bordered surface. `key` must start with 'card' so the stylesheet can target it."""
    return st.container(key=key)


def card_title(title: str, subtitle: str = None):
    sub = f"<span>{esc(subtitle)}</span>" if subtitle else ""
    ui(f'<div class="vl-card-title"><h3>{esc(title)}</h3>{sub}</div>')


def stat_tiles(items: list, cols: int = 4):
    """items: list of (label_html, value, sub, small) tuples; label is trusted HTML (for tooltips)."""
    tiles = []
    for label, value, sub, small in items:
        sub_html = f'<div class="s">{esc(sub)}</div>' if sub else ""
        cls = "v sm" if small else "v"
        tiles.append(f'<div class="vl-stat"><div class="k">{label}</div><div class="{cls}">{esc(value)}</div>{sub_html}</div>')
    ui(f'<div class="vl-grid c{cols}">{"".join(tiles)}</div>')


def style_fig(fig, height: int, title: str = None, legend: bool = True):
    """Shared, recessive chart styling: Inter, light grid, no chart junk, compact margins."""
    fig.update_layout(
        height=height,
        template="plotly_white",
        font=dict(family=FONT_STACK, size=12, color=INK["secondary"]),
        title=dict(text=title or "", x=0, xanchor="left", y=1, yref="container", yanchor="top",
                   pad=dict(t=6), font=dict(size=14, color=INK["primary"])),
        margin=dict(l=8, r=16, t=(78 if legend else 48) if title else (40 if legend else 12), b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(color=INK["secondary"])),
        hoverlabel=dict(bgcolor="white", bordercolor=INK["border"], font=dict(family=FONT_STACK, color=INK["primary"])),
    )
    fig.update_xaxes(gridcolor=INK["grid"], zeroline=False, linecolor=INK["border"], tickfont=dict(color=INK["muted"]))
    fig.update_yaxes(gridcolor=INK["grid"], zeroline=False, linecolor=INK["border"], tickfont=dict(color=INK["secondary"]))
    return fig


def goto(section_key: str):
    """on_click callback used by in-page buttons to jump to another section."""
    st.session_state["nav"] = section_key


# ======================================================================================
# 6. SIDEBAR
# ======================================================================================

def render_sidebar():
    with st.sidebar:
        ui(
            '<div class="vl-brand"><div class="vl-logo">'
            '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.2" '
            'stroke-linecap="round"><circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l5 5"/>'
            '<path d="M8 10.5h5M8 8h3" stroke-width="1.8"/></svg></div>'
            '<div><div class="vl-brand-title">KGIRS Virtual Lab</div>'
            '<div class="vl-brand-sub">BM25 Ranking &middot; Exp. 5</div></div></div>'
        )
        section = st.radio(
            "Navigation", NAV_KEYS, key="nav",
            format_func=lambda k: NAV_LABELS[k], label_visibility="collapsed"
        )

        with st.container(key="sb_progress"):
            slot = st.empty()
    return section, slot


def render_progress(slot):
    """Filled at the end of the run so it reflects actions taken on the current page (e.g. a new trial)."""
    trials = len(st.session_state.get("trials", []))
    quiz_done = st.session_state.get("quiz_submitted", False)
    score = st.session_state.get("quiz_score", 0)
    eligible = trials > 0 and quiz_done
    done_count = int(trials > 0) + int(quiz_done) + int(eligible)

    def step(done, text, meta=""):
        meta_html = f"<em>{esc(meta)}</em>" if meta else ""
        icon = ICON_CHECK if done else ""
        return f'<li class="{"done" if done else ""}"><span class="dot">{icon}</span>{esc(text)}{meta_html}</li>'

    with slot:
        ui(
            '<div class="vl-progress">'
            f'<div class="vl-progress-head"><span>Your progress</span><span>{done_count} / 3</span></div>'
            f'<div class="vl-bar"><i style="width:{done_count / 3 * 100:.0f}%"></i></div>'
            '<ul class="vl-steps">'
            + step(trials > 0, "Simulation trial", f"{trials} recorded" if trials else "")
            + step(quiz_done, "Quiz submitted", f"{score}/{len(QUIZ_QUESTIONS)}" if quiz_done else "")
            + step(eligible, "Certificate unlocked")
            + "</ul></div>"
        )


def scroll_to_top(section_key: str):
    """Resets the main scroll position when the user switches sections (script only changes then)."""
    with st.container(key="vl_scroll"):
        st.html(
            f'<span data-section="{esc(section_key)}"></span><script>'
            '(function(){var m=document.querySelector(\'[data-testid="stMain"]\');'
            'if(m){m.scrollTo({top:0,behavior:"instant"});}window.scrollTo(0,0);})();</script>',
            unsafe_allow_javascript=True,
        )


# ======================================================================================
# 7. SECTION PAGES
# ======================================================================================

def render_purpose_section():
    page_header("Purpose", "What this experiment sets out to do, and what you will be able to do by the end of it.")

    with card("card_aim"):
        ui(
            '<div class="vl-aim"><div class="vl-aim-mark">'
            '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" '
            'stroke-linecap="round"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/>'
            '<circle cx="12" cy="12" r="0.8" fill="currentColor"/></svg></div>'
            '<div><div class="k">Aim</div><p>To implement the BM25 probabilistic ranking function and empirically '
            'compare its document ranking behaviour against the classic TF-IDF vector-space model '
            '(cosine similarity).</p></div></div>'
        )

    ui('<div class="vl-h2">Learning objectives</div>')
    titles = ["Understand the baseline", "Implement BM25", "Compare the rankings", "Analyse the parameters"]
    tiles = "".join(
        f'<div class="vl-tile"><div class="num">0{i + 1}</div><h4>{esc(titles[i])}</h4><p>{esc(obj)}</p></div>'
        for i, obj in enumerate(EXPERIMENT_CONFIG["objectives"])
    )
    ui(f'<div class="vl-grid c2">{tiles}</div>')

    ui('<div class="vl-h2">Lab at a glance</div>')
    stat_tiles([
        ("Experiment", "No. 5", "KGIRS Virtual Lab", False),
        ("Models compared", "TF-IDF vs BM25", "Cosine vs probabilistic", True),
        ("Built-in corpus", f"{len(BUILTIN_CORPUS)} documents", "Or paste / upload your own", True),
        ("Assessment", f"{len(QUIZ_QUESTIONS)} questions", "Instant feedback", True),
    ])

    ui('<div class="vl-h2">How to complete this lab</div>')
    ui(
        '<div class="vl-flow">'
        '<div><b>Theory</b><span>Read the models, formulas and the ranking pipeline.</span></div>'
        '<div><b>Simulation</b><span>Run a query, watch the pipeline and record a trial.</span></div>'
        '<div><b>Quiz</b><span>Answer 10 conceptual questions.</span></div>'
        '<div><b>Report</b><span>Download your PDF lab report.</span></div>'
        '<div><b>Certificate</b><span>Generate your certificate of completion.</span></div>'
        '</div>'
    )
    st.write("")
    st.button("Start with Theory", type="primary", icon=":material/arrow_forward:",
              on_click=goto, args=("theory",))


def render_algorithm_diagram():
    """Flowchart of the ranking pipeline: shared preprocessing, then two parallel scoring paths."""
    fig = go.Figure()
    boxes = [
        {"id": "input", "x": 0.55, "y": 0, "w": 1.8, "h": 0.9, "text": "<b>Corpus + query</b><br>raw text",
         "fill": "#f8fafc", "line": "#cbd5e1"},
        {"id": "tok", "x": 2.75, "y": 0, "w": 1.9, "h": 0.9, "text": "<b>Tokenize</b><br>lowercase, stopwords",
         "fill": "#f8fafc", "line": "#cbd5e1"},
        {"id": "freq", "x": 4.95, "y": 0, "w": 2.0, "h": 0.9, "text": "<b>Count</b><br>tf, df, doc length",
         "fill": "#f8fafc", "line": "#cbd5e1"},
        {"id": "tfidf", "x": 7.4, "y": 1.05, "w": 2.4, "h": 0.9, "text": "<b>TF-IDF</b><br>cosine similarity",
         "fill": "#fdf1eb", "line": SERIES["tfidf"]},
        {"id": "bm25", "x": 7.4, "y": -1.05, "w": 2.4, "h": 0.9, "text": "<b>BM25</b><br>IDF x saturated tf (k1, b)",
         "fill": "#eaf2fc", "line": SERIES["bm25"]},
        {"id": "rank", "x": 10.0, "y": 0, "w": 1.9, "h": 0.9, "text": "<b>Rank</b><br>sort by score",
         "fill": "#eff6ff", "line": "#2563eb"},
    ]
    for bx in boxes:
        fig.add_shape(type="rect", x0=bx["x"] - bx["w"] / 2, x1=bx["x"] + bx["w"] / 2,
                      y0=bx["y"] - bx["h"] / 2, y1=bx["y"] + bx["h"] / 2,
                      line=dict(color=bx["line"], width=1.5), fillcolor=bx["fill"], layer="below")
        fig.add_annotation(x=bx["x"], y=bx["y"], text=bx["text"], showarrow=False, align="center",
                           font=dict(size=12, color=INK["primary"], family=FONT_STACK))
    by_id = {bx["id"]: bx for bx in boxes}
    for src, dst in [("input", "tok"), ("tok", "freq"), ("freq", "tfidf"), ("freq", "bm25"),
                     ("tfidf", "rank"), ("bm25", "rank")]:
        a, c = by_id[src], by_id[dst]
        y0 = a["y"] if a["y"] == c["y"] else a["y"] + (0.2 if c["y"] > a["y"] else -0.2)
        y1 = c["y"]
        fig.add_annotation(x=c["x"] - c["w"] / 2, y=y1, ax=a["x"] + a["w"] / 2, ay=y0,
                           xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True,
                           arrowhead=2, arrowsize=1.1, arrowwidth=1.5, arrowcolor="#94a3b8")
    fig.update_xaxes(visible=False, range=[-0.5, 11.2])
    fig.update_yaxes(visible=False, range=[-1.75, 1.75], scaleanchor="x", scaleratio=1)
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    st.plotly_chart(fig, width="stretch", config={"staticPlot": True, "displayModeBar": False})


def render_theory_section():
    page_header("Theory", "The two ranking models behind this experiment: TF-IDF with cosine similarity, "
                          "and the BM25 probabilistic ranking function.")

    with card("card_background"):
        st.markdown(THEORY_CONTENT["background"], unsafe_allow_html=True)
        st.caption("Tip: terms with a dotted underline (TF-IDF, BM25, k1, b ...) show a short definition on hover.")

    with card("card_compare"):
        card_title("TF-IDF vs. BM25 at a glance")
        rows = "".join(f"<tr><td>{esc(a)}</td><td>{esc(t)}</td><td>{esc(b_)}</td></tr>" for a, t, b_ in COMPARISON_TABLE)
        ui(f'<table class="vl-table"><thead><tr><th>Aspect</th><th>TF-IDF (cosine similarity)</th>'
                f'<th>BM25</th></tr></thead><tbody>{rows}</tbody></table>')

    with card("card_pipeline"):
        card_title("How the algorithm works", "Both models share the same preprocessing, then score documents "
                                              "along two parallel paths before ranking.")
        render_algorithm_diagram()

    ui('<div class="vl-h2">Applications of BM25</div>'
            '<p class="vl-lead">BM25 is the working relevance engine behind many real-world retrieval systems.</p>')
    apps = "".join(f'<div class="vl-tile"><h4>{esc(name)}</h4><p>{esc(desc)}</p></div>' for name, desc in BM25_APPLICATIONS)
    ui(f'<div class="vl-grid c3">{apps}</div>')

    st.write("")
    with card("card_procedure"):
        card_title("Experimental procedure")
        items = "".join(f"<li>{esc(step_)}</li>" for step_ in THEORY_CONTENT["procedure"])
        ui(f'<ol class="vl-steps-list">{items}</ol>')

    with st.expander("Key terminology & variable reference", icon=":material/dictionary:"):
        rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in THEORY_CONTENT["key_terms"].items())
        ui(f'<table class="vl-table"><thead><tr><th>Term / variable</th><th>Definition & role</th></tr></thead>'
                f'<tbody>{rows}</tbody></table>')


def render_references_section():
    page_header("References", "The foundational papers and texts this virtual lab draws on.")
    for i, ref in enumerate(REFERENCES_CONTENT, 1):
        ui(
            f'<div class="vl-ref"><div class="idx">{i}</div><div><p>{esc(ref["citation"])}</p>'
            f'<a href="{esc(ref["url"])}" target="_blank" rel="noopener noreferrer">Open source {ICON_ARROW}</a></div></div>'
        )


# ---------------------------------------------------------------------------------------
# Simulation: live, auto-playing pipeline animation (runs client-side in a sandboxed iframe)
# ---------------------------------------------------------------------------------------

SIM_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--ink:#0f172a;--ink2:#334155;--muted:#64748b;--border:#e2e8f0;--surface:#f8fafc;--primary:#2563eb;--p50:#eff6ff;
--bm25:#2a78d6;--tfidf:#eb6834;--up:#15803d;--down:#b91c1c}
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{font-family:Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:var(--ink);background:#fff;-webkit-font-smoothing:antialiased}
.sim{display:flex;flex-direction:column;height:100%;gap:12px;padding:1px}
.top{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}
.steps{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.step{display:flex;align-items:center;gap:7px;padding:5px 12px 5px 5px;border-radius:999px;border:1px solid var(--border);
background:#fff;color:var(--muted);font-size:12.5px;font-weight:500;transition:all .3s ease}
.step .n{width:22px;height:22px;border-radius:50%;display:grid;place-items:center;background:var(--surface);font-size:11.5px;font-weight:600;transition:all .3s}
.step.active{border-color:#93c5fd;color:#1d4ed8;background:var(--p50)}
.step.active .n{background:var(--primary);color:#fff;box-shadow:0 0 0 4px rgba(37,99,235,.15)}
.step.done{color:var(--ink2)}.step.done .n{background:#dbeafe;color:#1d4ed8}
.sep{width:12px;height:1px;background:var(--border)}
.ctrls{display:flex;gap:8px;align-items:center}
.seg{display:flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.seg button{border:0;background:#fff;padding:6px 11px;font:500 12.5px Inter,sans-serif;color:var(--muted);cursor:pointer;transition:all .15s}
.seg button.on{background:var(--ink);color:#fff}
.seg button:disabled{cursor:not-allowed;opacity:.45}
.btn{border:1px solid var(--border);background:#fff;border-radius:8px;padding:6px 12px;font:600 12.5px Inter,sans-serif;color:var(--ink2);
cursor:pointer;display:flex;gap:6px;align-items:center;transition:background .15s}
.btn:hover{background:var(--surface)}
.body{flex:1;display:grid;grid-template-columns:262px minmax(0,1fr);gap:12px;min-height:0}
.panel{border:1px solid var(--border);border-radius:12px;background:var(--surface);padding:14px;display:flex;flex-direction:column;gap:10px;min-height:0;overflow:hidden}
.lbl{font-size:10.5px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}
.raw{font-size:13.5px;font-weight:500;background:#fff;border:1px solid var(--border);border-radius:8px;padding:8px 10px;word-break:break-word}
.tokens{display:flex;flex-wrap:wrap;gap:6px;min-height:26px}
.tok{font-size:12px;padding:3px 9px;border-radius:999px;background:#fff;border:1px solid var(--border);color:var(--ink2);
opacity:0;transform:translateY(4px);transition:all .35s ease}
.tok.in{opacity:1;transform:none}
.tok.keep{background:var(--p50);border-color:#bfdbfe;color:#1d4ed8;font-weight:600}
.tok.drop{text-decoration:line-through;opacity:.4}
.narr{font-size:12.8px;line-height:1.55;color:var(--ink2);transition:opacity .25s ease}
.narr b{color:var(--ink);font-weight:650}
.kv{display:grid;grid-template-columns:1fr auto;gap:5px 10px;font-size:12px;margin-top:auto;border-top:1px solid var(--border);padding-top:10px}
.kv span:nth-child(odd){color:var(--muted)}.kv span:nth-child(even){font-weight:600;text-align:right;font-variant-numeric:tabular-nums}
.listwrap{border:1px solid var(--border);border-radius:12px;display:flex;flex-direction:column;min-height:0;background:#fff;overflow:hidden}
.cols{display:grid;grid-template-columns:40px minmax(0,1fr) 190px 60px;gap:10px;align-items:center}
.lhead{padding:9px 17px;border-bottom:1px solid var(--border);font-size:10.5px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);background:var(--surface)}
.legend{display:flex;gap:10px;text-transform:none;letter-spacing:0;font-weight:500;font-size:11.5px;color:var(--ink2)}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px}
.lhead .r{text-align:right}
.list{position:relative;overflow-y:auto;flex:1;padding:4px 6px 6px}
.row{padding:5px 8px;border-radius:8px;border-left:3px solid transparent;opacity:0;transform:translateY(6px);
transition:opacity .35s ease,transform .35s ease,background-color .25s ease,border-color .25s ease}
.row.in{opacity:1;transform:none}
.row.scan{background:#fff7ed;border-left-color:var(--tfidf)}
.row.miss{opacity:.4}
.row.top{background:var(--p50);border-left-color:var(--primary)}
.rank{font-size:12px;font-weight:600;color:var(--muted);font-variant-numeric:tabular-nums;text-align:center;background:var(--surface);border-radius:6px;padding:3px 0;transition:all .25s}
.row.top .rank{background:var(--primary);color:#fff}
.main{min-width:0}
.title{font-size:13px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.meta{display:flex;gap:5px;align-items:center;margin-top:2px;min-height:17px;overflow:hidden}
.len{font-size:11px;color:var(--muted);white-space:nowrap}
.chip{font-size:10.5px;padding:1px 7px;border-radius:999px;background:#eef2ff;color:#1d4ed8;font-weight:600;white-space:nowrap;animation:pop .3s ease both}
@keyframes pop{from{opacity:0;transform:scale(.7)}to{opacity:1;transform:none}}
.bars{display:flex;flex-direction:column;gap:4px}
.bar{display:grid;grid-template-columns:1fr 42px;gap:6px;align-items:center}
.track{height:6px;border-radius:3px;background:#f1f5f9;overflow:hidden}
.fill{height:100%;width:0;border-radius:3px;transition:width .9s cubic-bezier(.2,.8,.2,1)}
.bar.b .fill{background:var(--bm25)}.bar.t .fill{background:var(--tfidf)}
.val{font-size:11px;font-variant-numeric:tabular-nums;color:var(--ink2);text-align:right}
.delta{font-size:11.5px;font-weight:600;text-align:right;font-variant-numeric:tabular-nums;opacity:0;transition:opacity .4s;white-space:nowrap}
.delta.show{opacity:1}.delta.up{color:var(--up)}.delta.down{color:var(--down)}.delta.same{color:var(--muted)}
.lhead .r{white-space:nowrap}
@media (max-width:900px){.body{grid-template-columns:205px minmax(0,1fr)}.cols{grid-template-columns:34px minmax(0,1fr) 128px 52px}
.narr{font-size:12px}.panel{padding:12px}}
@media (max-width:600px){.body{grid-template-columns:1fr;grid-template-rows:auto 1fr}.panel{max-height:170px}
.cols{grid-template-columns:30px minmax(0,1fr) 96px 40px}.narr,.kv{display:none}.meta .chip{display:none}}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style></head><body>
<div class="sim">
  <div class="top">
    <div class="steps" id="steps">
      <div class="step"><span class="n">1</span>Tokenize</div><div class="sep"></div>
      <div class="step"><span class="n">2</span>Match</div><div class="sep"></div>
      <div class="step"><span class="n">3</span>Score</div><div class="sep"></div>
      <div class="step"><span class="n">4</span>Rank</div>
    </div>
    <div class="ctrls">
      <div class="seg" title="Order the final list by">
        <button id="oB" class="on" disabled>BM25 order</button><button id="oT" disabled>TF-IDF order</button>
      </div>
      <button class="btn" id="replay"><svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 8a5.5 5.5 0 1 0 1.7-4"/><path d="M2.5 2.5v3h3"/></svg>Replay</button>
    </div>
  </div>
  <div class="body">
    <aside class="panel">
      <div class="lbl">Query</div>
      <div class="raw" id="raw"></div>
      <div class="lbl">Tokens</div>
      <div class="tokens" id="tokens"></div>
      <div class="narr" id="narr"></div>
      <div class="kv">
        <span>Documents</span><span id="sN"></span><span>Matched</span><span id="sM">&ndash;</span>
        <span>avgdl</span><span id="sA"></span><span>k1 / b</span><span id="sK"></span>
      </div>
    </aside>
    <section class="listwrap">
      <div class="lhead cols"><span>#</span><span>Document</span>
        <span class="legend"><span><i style="background:var(--bm25)"></i>BM25</span><span><i style="background:var(--tfidf)"></i>TF-IDF</span></span>
        <span class="r" id="dHead">&Delta; rank</span></div>
      <div class="list" id="list"></div>
    </section>
  </div>
</div>
<script>
const D = __PAYLOAD__;
const $ = s => document.querySelector(s);
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const T = ms => reduce ? 0 : ms;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const maxB = Math.max(0, ...D.docs.map(d => d.bm25));
const maxT = Math.max(0, ...D.docs.map(d => d.tfidf));
let runId = 0, mode = 'bm25';

function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
function setStep(n) { document.querySelectorAll('.step').forEach((s, i) => { s.classList.toggle('active', i + 1 === n); s.classList.toggle('done', i + 1 < n); }); }
function narrate(title, text) {
  const n = $('#narr'); n.style.opacity = 0;
  setTimeout(() => { n.innerHTML = ''; const b = el('b', null, title); n.append(b, document.createTextNode(' ' + text)); n.style.opacity = 1; }, T(180));
}
function countUp(node, target) {
  if (reduce) { node.textContent = target.toFixed(3); return; }
  const t0 = performance.now(), dur = 850;
  (function f(t) { const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3); node.textContent = (target * e).toFixed(3); if (p < 1) requestAnimationFrame(f); })(t0);
}
function setButtons(on) { $('#oB').disabled = !on; $('#oT').disabled = !on; $('#oB').classList.toggle('on', mode === 'bm25'); $('#oT').classList.toggle('on', mode === 'tfidf'); }

function build() {
  $('#raw').textContent = D.query;
  const tk = $('#tokens'); tk.innerHTML = '';
  D.tokens.forEach(t => { const c = el('span', 'tok', t.t); c.dataset.keep = t.kept ? '1' : '0'; tk.appendChild(c); });
  const list = $('#list'); list.innerHTML = ''; list.scrollTop = 0;
  D.docs.forEach((d, i) => {
    const r = el('div', 'row cols'); r._d = d;
    const main = el('div', 'main'); const ti = el('div', 'title', d.title); ti.title = d.title;
    const meta = el('div', 'meta'); meta.appendChild(el('span', 'len', d.len + ' tokens')); main.append(ti, meta);
    const bars = el('div', 'bars');
    ['b', 't'].forEach(k => { const bar = el('div', 'bar ' + k), tr = el('div', 'track'); tr.appendChild(el('div', 'fill')); bar.append(tr, el('span', 'val', '–')); bars.appendChild(bar); });
    r.append(el('div', 'rank', 'D' + (i + 1)), main, bars, el('div', 'delta', ''));
    list.appendChild(r);
  });
  $('#sN').textContent = D.N; $('#sM').textContent = '–'; $('#sA').textContent = D.avgdl; $('#sK').textContent = D.k1 + ' / ' + D.b;
  $('#dHead').textContent = 'Δ rank'; mode = 'bm25'; setButtons(false); setStep(0);
}

function reorder(m, animate) {
  mode = m; const list = $('#list'); const rows = [...list.children];
  const first = new Map(rows.map(r => [r, r.getBoundingClientRect().top]));
  const key = m === 'bm25' ? 'r_bm25' : 'r_tfidf', other = m === 'bm25' ? 'r_tfidf' : 'r_bm25';
  const scoreKey = m === 'bm25' ? 'bm25' : 'tfidf';
  rows.sort((a, b) => a._d[key] - b._d[key]).forEach(r => list.appendChild(r));
  rows.forEach(r => {
    const d = r._d, rk = d[key], mv = d[other] - rk;
    r.querySelector('.rank').textContent = rk;
    r.classList.toggle('top', rk <= D.topk && d[scoreKey] > 0);
    const dl = r.querySelector('.delta');
    dl.className = 'delta show ' + (mv > 0 ? 'up' : mv < 0 ? 'down' : 'same');
    dl.textContent = mv > 0 ? '▲ ' + mv : mv < 0 ? '▼ ' + (-mv) : '–';
    dl.title = 'Rank ' + rk + ' by ' + (m === 'bm25' ? 'BM25' : 'TF-IDF') + ', rank ' + d[other] + ' by ' + (m === 'bm25' ? 'TF-IDF' : 'BM25');
  });
  $('#dHead').textContent = m === 'bm25' ? 'vs TF-IDF' : 'vs BM25';
  if (animate && !reduce) {
    rows.forEach(r => { const dy = first.get(r) - r.getBoundingClientRect().top; if (dy) { r.style.transition = 'none'; r.style.transform = 'translateY(' + dy + 'px)'; } });
    void list.offsetHeight;
    rows.forEach(r => { r.style.transition = 'transform .75s cubic-bezier(.2,.8,.2,1), background-color .25s, border-color .25s, opacity .35s'; r.style.transform = ''; });
  }
  list.scrollTo({ top: 0, behavior: reduce ? 'auto' : 'smooth' });
  setButtons(true);
}

async function run() {
  const id = ++runId, alive = () => id === runId;
  build();
  const rows = [...document.querySelectorAll('.row')], toks = [...document.querySelectorAll('.tok')];

  setStep(1); narrate('Tokenize.', D.text.s1);
  rows.forEach((r, i) => setTimeout(() => alive() && r.classList.add('in'), T(30 * i)));
  await sleep(T(400));
  for (const t of toks) { if (!alive()) return; t.classList.add('in'); await sleep(T(150)); }
  await sleep(T(350)); if (!alive()) return;
  toks.forEach(t => t.classList.add(t.dataset.keep === '1' ? 'keep' : 'drop'));
  await sleep(T(1100)); if (!alive()) return;

  setStep(2); narrate('Match.', D.text.s2);
  let matched = 0; const per = Math.max(40, Math.min(115, 1700 / Math.max(1, rows.length)));
  for (const r of rows) {
    if (!alive()) return;
    r.classList.add('scan');
    if (!reduce) { const L = $('#list'), top = r.offsetTop, bot = top + r.offsetHeight;
      if (bot > L.scrollTop + L.clientHeight) L.scrollTo({ top: bot - L.clientHeight + 6, behavior: 'smooth' });
      else if (top < L.scrollTop) L.scrollTo({ top: Math.max(0, top - 6), behavior: 'smooth' }); }
    await sleep(T(per));
    r.classList.remove('scan');
    const d = r._d, hits = D.terms.filter(t => (d.tf[t] || 0) > 0);
    if (hits.length) { matched++; const meta = r.querySelector('.meta'); hits.forEach(t => meta.appendChild(el('span', 'chip', t + ' ×' + d.tf[t]))); }
    else r.classList.add('miss');
    $('#sM').textContent = matched + ' / ' + D.N;
  }
  $('#list').scrollTo({ top: 0, behavior: reduce ? 'auto' : 'smooth' });
  await sleep(T(600)); if (!alive()) return;

  setStep(3); narrate('Score.', D.text.s3);
  rows.forEach((r, i) => setTimeout(() => {
    if (!alive()) return;
    const d = r._d, f = r.querySelectorAll('.fill'), v = r.querySelectorAll('.val');
    f[0].style.width = (maxB > 0 ? d.bm25 / maxB * 100 : 0) + '%';
    f[1].style.width = (maxT > 0 ? d.tfidf / maxT * 100 : 0) + '%';
    countUp(v[0], d.bm25); countUp(v[1], d.tfidf);
  }, T(35 * i)));
  await sleep(T(1500 + 35 * rows.length)); if (!alive()) return;

  setStep(4); narrate('Rank.', D.text.s4);
  reorder('bm25', true);
  await sleep(T(1100)); if (!alive()) return;
  setStep(5); narrate('Done.', D.text.done);
}

$('#replay').onclick = () => run();
$('#oB').onclick = () => { if (mode !== 'bm25') reorder('bm25', true); };
$('#oT').onclick = () => { if (mode !== 'tfidf') reorder('tfidf', true); };

// Start automatically as soon as the simulation scrolls into view.
build();
let started = false;
const io = new IntersectionObserver(es => { if (!started && es.some(e => e.isIntersecting)) { started = true; io.disconnect(); run(); } }, { threshold: 0.25 });
io.observe(document.querySelector('.sim'));
setTimeout(() => { if (!started) { started = true; io.disconnect(); run(); } }, 2500);
</script></body></html>"""


def build_simulation_payload(documents, query, k1, b, ranking_output, bm25_sorted, tfidf_sorted, top_k) -> dict:
    raw_tokens = re.findall(r"[a-zA-Z]+", query.lower())
    terms = ranking_output["unique_query_terms"]
    r_bm25 = {r["id"]: i + 1 for i, r in enumerate(bm25_sorted)}
    r_tfidf = {r["id"]: i + 1 for i, r in enumerate(tfidf_sorted)}
    by_id = {r["id"]: r for r in ranking_output["results"]}
    docs = []
    for doc in documents:
        counts = collections.Counter(tokenize(doc["text"]))
        res = by_id[doc["id"]]
        docs.append({
            "id": doc["id"], "title": doc["title"], "len": res["length"],
            "tf": {t: counts.get(t, 0) for t in terms},
            "bm25": res["bm25_score"], "tfidf": res["tfidf_score"],
            "r_bm25": r_bm25[doc["id"]], "r_tfidf": r_tfidf[doc["id"]],
        })
    top_b = bm25_sorted[0]["title"] if bm25_sorted else "-"
    top_t = tfidf_sorted[0]["title"] if tfidf_sorted else "-"
    kept = [t for t in raw_tokens if t not in STOPWORDS and len(t) > 1]
    s1 = ("The query is lower-cased and split into words. Stopwords such as “the” or “and” carry "
          "little meaning, so they are dropped; the remaining words are the query terms.")
    if not kept:
        s1 += " No query terms remain — try a more specific query."
    return {
        "query": query, "terms": terms, "N": len(documents), "avgdl": ranking_output["avgdl"],
        "k1": k1, "b": b, "topk": top_k,
        "tokens": [{"t": t, "kept": (t not in STOPWORDS and len(t) > 1)} for t in raw_tokens],
        "docs": docs,
        "text": {
            "s1": s1,
            "s2": "Every document is scanned for the query terms and their term frequency (tf) is counted. "
                  "Documents with no matching term fade out — they score zero.",
            "s3": f"TF-IDF measures the cosine similarity between TF-IDF vectors. BM25 sums IDF × saturated tf "
                  f"per term, normalised by document length (k1 = {k1}, b = {b}, avgdl = {ranking_output['avgdl']}). "
                  f"Bars are scaled to each method’s top score.",
            "s4": f"Documents are sorted by BM25 score and the top {top_k} are retrieved (highlighted). Arrows show how "
                  f"far each document moved compared with its TF-IDF rank.",
            "done": f"BM25 ranks “{top_b}” first; TF-IDF ranks “{top_t}” first. Switch the order "
                    f"above to compare the two lists, or press Replay.",
        },
    }


def render_live_simulation(payload: dict, height: int = 600):
    data = json.dumps(payload).replace("</", "<\\/")
    components.html(SIM_TEMPLATE.replace("__PAYLOAD__", data), height=height, scrolling=False)


def render_simulation_section():
    page_header("Simulation", "Configure a corpus and a query. The ranking pipeline plays live, then you can "
                              "drill into the results and record a trial.")

    # ---------------- Setup ----------------
    with card("card_setup"):
        card_title("Setup", "Choose the documents, the query, and the BM25 parameters.")
        source = st.segmented_control("Document source", SOURCE_OPTIONS, key="sim_source", required=True)

        if source == "Paste documents":
            st.text_area("Documents (separate each document with a blank line)", key="paste_text", height=170,
                         placeholder="Document one text goes here...\n\nDocument two text goes here...")
            documents = parse_custom_documents(st.session_state.get("paste_text", ""))
            query = st.text_input("Search query", key="q_paste", placeholder="e.g. neural networks")
        elif source == "Upload PDF / DOCX":
            uploaded_files = st.file_uploader("Upload two or more PDF or DOCX files - each file becomes one document",
                                              type=["pdf", "docx"], accept_multiple_files=True, key="uploads")
            documents = []
            for i, uf in enumerate(uploaded_files or []):
                text = extract_text_from_upload(uf)
                if text:
                    documents.append({"id": i + 1, "title": uf.name, "text": text})
                else:
                    st.warning(f"Could not extract text from **{uf.name}** (empty, image-only, or unsupported).")
            if documents:
                with st.expander(f"Preview extracted text ({len(documents)} documents)"):
                    for doc in documents:
                        st.markdown(f"**{doc['id']}. {doc['title']}**")
                        st.caption(doc["text"][:400] + ("..." if len(doc["text"]) > 400 else ""))
            query = st.text_input("Search query", key="q_upload", placeholder="e.g. information retrieval")
        else:
            documents = BUILTIN_CORPUS
            with st.expander(f"View the built-in corpus ({len(BUILTIN_CORPUS)} documents)"):
                for doc in documents:
                    st.markdown(f"**{doc['id']}. {doc['title']}** — {doc['text']}")
            query = st.text_input("Search query", key="q_builtin")

        cfg = SIMULATION_CONFIG
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            k1 = st.slider("k1 · term-frequency saturation", min_value=cfg["k1_min"], max_value=cfg["k1_max"],
                           step=cfg["k1_step"], key="k1", help=GLOSSARY["k1"])
        with c2:
            b = st.slider("b · document-length normalization", min_value=cfg["b_min"], max_value=cfg["b_max"],
                          step=cfg["b_step"], key="b", help=GLOSSARY["b"])

    if len(documents) < 2:
        st.info("Add at least **2 documents** above to run the simulation.", icon=":material/info:")
        return
    if not (query or "").strip():
        st.info("Enter a **search query** above to run the simulation.", icon=":material/info:")
        return

    ranking_output = rank_documents(documents, query, k1, b)
    results = ranking_output["results"]
    tfidf_sorted = sorted(results, key=lambda r: r["tfidf_score"], reverse=True)
    bm25_sorted = sorted(results, key=lambda r: r["bm25_score"], reverse=True)
    correlation = spearman_rank_correlation([r["id"] for r in tfidf_sorted], [r["id"] for r in bm25_sorted])
    top_k = min(5, len(bm25_sorted))
    query_terms = ranking_output["unique_query_terms"]

    # ---------------- Headline numbers ----------------
    st.write("")
    stat_tiles([
        ("Top by BM25", bm25_sorted[0]["title"], f"score {bm25_sorted[0]['bm25_score']:.3f}", True),
        ("Top by TF-IDF", tfidf_sorted[0]["title"], f"score {tfidf_sorted[0]['tfidf_score']:.3f}", True),
        (gterm("avgdl"), f"{ranking_output['avgdl']}", "average tokens per document", False),
        (gterm("Rank correlation (ρ)", "rank correlation"), f"{correlation}", "1.0 = identical orderings", False),
    ])

    # ---------------- Live simulation ----------------
    st.write("")
    with card("card_live"):
        card_title("Live retrieval simulation", "Plays automatically whenever the query or parameters change: "
                                                "tokenize → match → score → rank. Use Replay to watch again.")
        payload = build_simulation_payload(documents, query, k1, b, ranking_output, bm25_sorted, tfidf_sorted, top_k)
        render_live_simulation(payload, height=600)

    # ---------------- Analysis ----------------
    with card("card_analysis"):
        card_title("Results & analysis")
        t_rank, t_score, t_terms, t_heat, t_funnel, t_embed = st.tabs(
            ["Rankings", "Score comparison", "Term contributions", "Term heatmap", "Retrieval funnel", "Embedding space"])

        with t_rank:
            r_tfidf = {r["id"]: i + 1 for i, r in enumerate(tfidf_sorted)}
            max_b = max(r["bm25_score"] for r in results) or 1.0
            max_t = max(r["tfidf_score"] for r in results) or 1.0
            ca, cb = st.columns(2, gap="medium")
            with ca:
                st.markdown("**BM25 ranking**")
                st.dataframe(
                    pd.DataFrame([{"Rank": i + 1, "Document": r["title"], "Score": r["bm25_score"],
                                   "vs TF-IDF": (lambda mv: f"▲ {mv}" if mv > 0 else (f"▼ {-mv}" if mv < 0 else "–"))(r_tfidf[r["id"]] - (i + 1)),
                                   "Length": r["length"]} for i, r in enumerate(bm25_sorted)]),
                    hide_index=True, width="stretch",
                    column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=float(max_b), format="%.3f"),
                                   "Rank": st.column_config.NumberColumn(width="small")})
            with cb:
                st.markdown("**TF-IDF ranking**")
                st.dataframe(
                    pd.DataFrame([{"Rank": i + 1, "Document": r["title"], "Score": r["tfidf_score"], "Length": r["length"]}
                                  for i, r in enumerate(tfidf_sorted)]),
                    hide_index=True, width="stretch",
                    column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=float(max_t), format="%.3f"),
                                   "Rank": st.column_config.NumberColumn(width="small")})

        with t_score:
            shown = bm25_sorted[:min(14, len(bm25_sorted))]
            titles = [r["title"] for r in shown]
            fig = go.Figure()
            fig.add_bar(y=titles, x=[r["bm25_score"] / max_b * 100 for r in shown], orientation="h", name="BM25",
                        marker=dict(color=SERIES["bm25"]), customdata=[r["bm25_score"] for r in shown],
                        hovertemplate="<b>%{y}</b><br>BM25 score %{customdata:.3f} (%{x:.0f}% of top)<extra></extra>")
            fig.add_bar(y=titles, x=[r["tfidf_score"] / max_t * 100 for r in shown], orientation="h", name="TF-IDF",
                        marker=dict(color=SERIES["tfidf"]), customdata=[r["tfidf_score"] for r in shown],
                        hovertemplate="<b>%{y}</b><br>TF-IDF score %{customdata:.3f} (%{x:.0f}% of top)<extra></extra>")
            fig.update_layout(barmode="group", bargap=0.32, bargroupgap=0.12, barcornerradius=4)
            style_fig(fig, height=90 + 38 * len(shown),
                      title="Relative score by document (each method scaled to its top result = 100)")
            fig.update_yaxes(autorange="reversed", showgrid=False)
            fig.update_xaxes(range=[0, 105], title_text="% of the method's top score", title_font=dict(color=INK["muted"]))
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
            st.caption("Scores are indexed per method because cosine similarity (0–1) and BM25 use different scales. "
                       "Hover a bar for the raw score. Documents are ordered by BM25.")

        with t_terms:
            top_doc = bm25_sorted[0]
            contributions = top_doc["term_contributions"]
            if contributions:
                items = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
                fig = go.Figure(go.Bar(
                    y=[k for k, _ in items], x=[v for _, v in items], orientation="h",
                    marker=dict(color=SERIES["bm25"]), text=[f"{v:.3f}" for _, v in items], textposition="outside",
                    textfont=dict(color=INK["secondary"]), cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>contributes %{x:.3f} to the BM25 score<extra></extra>"))
                fig.update_layout(barcornerradius=4, bargap=0.45)
                style_fig(fig, height=110 + 46 * len(items), title=f"Per-term BM25 contribution — {top_doc['title']}", legend=False)
                fig.update_yaxes(autorange="reversed", showgrid=False)
                st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
            else:
                st.info("None of the query terms appear in the top-ranked document.")

        with t_heat:
            if query_terms:
                doc_counts = [collections.Counter(tokenize(d["text"])) for d in documents]
                z = [[c.get(t, 0) for t in query_terms] for c in doc_counts]
                ylabels = [d["title"] for d in documents]
                fig = go.Figure(go.Heatmap(
                    z=z, x=query_terms, y=ylabels, xgap=2, ygap=2,
                    colorscale=[[0, "#f4f7fb"], [0.0001, "#dbe8f8"], [1, "#5b9be0"]],
                    text=[[str(v) if v else "" for v in row] for row in z], texttemplate="%{text}",
                    textfont=dict(color=INK["primary"], size=11),
                    colorbar=dict(title=dict(text="tf", font=dict(size=11)), thickness=10, outlinewidth=0),
                    hovertemplate="<b>%{y}</b><br>term “%{x}” appears %{z}×<extra></extra>"))
                style_fig(fig, height=max(300, 26 * len(documents) + 90),
                          title="Query-term frequency per document", legend=False)
                fig.update_yaxes(autorange="reversed", showgrid=False)
                fig.update_xaxes(side="top", showgrid=False)
                fig.update_layout(margin=dict(t=76))
                st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
            else:
                st.info("No query terms remain after tokenization.")

        with t_funnel:
            candidates = [r for r in results if r["term_contributions"]]
            fig = go.Figure(go.Funnel(
                y=["Documents in corpus", "Candidates (≥1 query term)", f"Top-{top_k} retrieved"],
                x=[len(documents), len(candidates), top_k], textinfo="value+percent initial",
                marker=dict(color=["#bcd4f2", "#6fa3e3", SERIES["bm25"]]),
                textfont=dict(color=INK["primary"], size=13), connector=dict(fillcolor="#eef2f7"),
                hovertemplate="%{y}: %{x}<extra></extra>"))
            style_fig(fig, height=330, title="Retrieval funnel", legend=False)
            fig.update_xaxes(showgrid=False, showticklabels=False)
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
            st.caption(f"Of **{len(documents)}** documents, **{len(candidates)}** contain at least one query term; "
                       f"the top **{top_k}** by BM25 are treated as retrieved.")

        with t_embed:
            retrieved_ids = {r["id"] for r in bm25_sorted[:top_k]}
            points = build_embedding_projection(documents, query, retrieved_ids)["points"]
            if points:
                q = next(p for p in points if p["id"] == "query")
                others = [p for p in points if p["id"] != "query" and not p["retrieved"]]
                hits = [p for p in points if p["id"] != "query" and p["retrieved"]]
                fig = go.Figure()
                for p in hits:
                    fig.add_trace(go.Scatter(x=[q["x"], p["x"]], y=[q["y"], p["y"]], mode="lines",
                                             line=dict(color="#94a3b8", dash="dot", width=1.2),
                                             showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=[p["x"] for p in others], y=[p["y"] for p in others], mode="markers",
                                         name="Other documents", text=[p["title"] for p in others],
                                         marker=dict(size=10, color=SERIES["neutral"], line=dict(width=2, color="white")),
                                         hovertemplate="%{text}<extra></extra>"))
                fig.add_trace(go.Scatter(x=[p["x"] for p in hits], y=[p["y"] for p in hits], mode="markers",
                                         name=f"Top-{top_k} retrieved (BM25)", text=[p["title"] for p in hits],
                                         marker=dict(size=13, color=SERIES["bm25"], line=dict(width=2, color="white")),
                                         hovertemplate="%{text}<extra></extra>"))
                fig.add_trace(go.Scatter(x=[q["x"]], y=[q["y"]], mode="markers", name="Query",
                                         marker=dict(size=18, symbol="star", color=SERIES["tfidf"], line=dict(width=2, color="white")),
                                         hovertemplate="Query<extra></extra>"))
                style_fig(fig, height=460, title="2-D projection of document and query TF-IDF vectors (SVD)")
                fig.update_xaxes(title_text="Component 1", title_font=dict(color=INK["muted"]))
                fig.update_yaxes(title_text="Component 2", title_font=dict(color=INK["muted"]))
                st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
                st.caption("Nearby points share vocabulary. Dotted lines connect the query to the documents BM25 retrieved.")
            else:
                st.info("Not enough vocabulary to build a projection yet.")

    # ---------------- Trial log ----------------
    with card("card_log"):
        card_title("Trial log book", "Record the current query, parameters and outcome. At least one trial is "
                                     "needed for the certificate.")
        b1, b2, _sp = st.columns([1.2, 1, 2.5])
        with b1:
            if st.button("Record current trial", type="primary", icon=":material/add:", width="stretch"):
                st.session_state["trials"].append({
                    "Trial #": len(st.session_state["trials"]) + 1, "Query": query, "k1": k1, "b": b,
                    "Top TF-IDF Doc": tfidf_sorted[0]["title"], "Top BM25 Doc": bm25_sorted[0]["title"],
                    "Rank Correlation": correlation, "Timestamp": datetime.now().strftime("%H:%M:%S"),
                })
                st.toast(f"Trial #{len(st.session_state['trials'])} recorded", icon=":material/check_circle:")
        with b2:
            if st.button("Clear log", icon=":material/delete:", width="stretch",
                         disabled=not st.session_state["trials"]):
                st.session_state["trials"] = []
                st.toast("Trial log cleared")
        if st.session_state["trials"]:
            df_trials = pd.DataFrame(st.session_state["trials"])
            st.dataframe(df_trials, width="stretch", hide_index=True)
            st.download_button("Download trials (.csv)", data=df_trials.to_csv(index=False).encode("utf-8"),
                               file_name="bm25_tfidf_trials.csv", mime="text/csv", icon=":material/download:")
        else:
            st.caption("No trials recorded yet.")


def render_quiz_section():
    page_header("Quiz", f"{len(QUIZ_QUESTIONS)} conceptual questions on TF-IDF and BM25. You get instant, "
                        f"explained feedback when you submit.")

    with st.form("lab_quiz_form", border=False):
        user_responses = {}
        for q in QUIZ_QUESTIONS:
            with st.container(key=f"card_q{q['id']}"):
                ui(f'<div class="vl-q">QUESTION {q["id"]} OF {len(QUIZ_QUESTIONS)}</div>'
                        f'<div class="vl-qt">{esc(q["question"])}</div>')
                selected = st.radio(f"Options for question {q['id']}", options=q["options"],
                                    index=st.session_state["quiz_answers"].get(q["id"]),
                                    key=f"quiz_radio_{q['id']}", label_visibility="collapsed")
                user_responses[q["id"]] = q["options"].index(selected) if selected is not None else None
        submitted = st.form_submit_button("Submit answers", type="primary", icon=":material/task_alt:")

    unanswered = [qid for qid, a in user_responses.items() if a is None]
    if submitted and unanswered:
        st.warning(f"Please answer every question before submitting — unanswered: "
                   f"{', '.join('Q' + str(q) for q in unanswered)}.", icon=":material/error:")
    elif submitted:
        st.session_state["quiz_answers"] = user_responses
        st.session_state["quiz_submitted"] = True
        st.session_state["quiz_score"] = sum(
            1 for q in QUIZ_QUESTIONS if user_responses.get(q["id"]) == q["answer_index"])
        st.rerun()

    if st.session_state.get("quiz_submitted"):
        score, total = st.session_state["quiz_score"], len(QUIZ_QUESTIONS)
        perc = score / total * 100
        verdict = "Excellent work." if perc >= 80 else ("Good attempt." if perc >= 50 else "Review the Theory section and try again.")
        with card("card_quiz_result"):
            ui(
                f'<div class="vl-score"><div class="big">{score}/{total}</div><div class="meta">'
                f'<b>{perc:.0f}% &middot; {esc(verdict)}</b>'
                f'<div class="vl-bar"><i style="width:{perc:.0f}%"></i></div>'
                f'<span style="font-size:.82rem;color:#64748b">Your score is saved to your report and certificate. '
                f'You can change answers and resubmit at any time.</span></div></div>'
            )
        fb = []
        for q in QUIZ_QUESTIONS:
            ua, ca = st.session_state["quiz_answers"].get(q["id"], 0), q["answer_index"]
            if ua == ca:
                fb.append(f'<div class="vl-fb ok"><div class="h"><i>Correct</i>Question {q["id"]}</div>'
                          f'<p>{esc(q["explanation"])}</p></div>')
            else:
                fb.append(f'<div class="vl-fb no"><div class="h"><i>Incorrect</i>Question {q["id"]}</div>'
                          f'<p>Your answer: {esc(q["options"][ua])}<br><b>Correct:</b> {esc(q["options"][ca])}</p>'
                          f'<p>{esc(q["explanation"])}</p></div>')
        with st.expander("Answer review & explanations", expanded=True, icon=":material/fact_check:"):
            ui("".join(fb))


def render_report_section():
    page_header("Report Generation", "Compile your details, recorded trials and quiz result into a downloadable "
                                     "PDF lab report.")

    with card("card_student"):
        card_title("Student details")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.text_input("Student name", key="rep_name")
        with c2:
            st.text_input("Roll number / ID", key="rep_id")
        with c3:
            st.date_input("Experiment date", key="rep_date")

    with card("card_notes"):
        card_title("Discussion & observations", "Your interpretation of the results. This appears in the report.")
        st.text_area("Observations", key="rep_notes", height=130, label_visibility="collapsed")

    student_name = st.session_state.get("rep_name", "")
    student_id = st.session_state.get("rep_id", "")
    lab_date = st.session_state.get("rep_date")
    notes = st.session_state.get("rep_notes", "")
    st.session_state["student_info"] = {"name": student_name, "id": student_id, "date": str(lab_date)}
    trials_df = pd.DataFrame(st.session_state["trials"]) if st.session_state["trials"] else pd.DataFrame()

    with card("card_summary"):
        card_title("Report preview")
        stat_tiles([
            ("Student", student_name or "—", student_id or "", True),
            ("Trials recorded", str(len(trials_df)), "from the Simulation", False),
            ("Quiz score", f"{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}",
             "submitted" if st.session_state.get("quiz_submitted") else "not submitted yet", False),
        ], cols=3)
        if not trials_df.empty:
            st.dataframe(trials_df, hide_index=True, width="stretch")
        else:
            st.caption("No trials recorded yet — the report will show 0 trials.")

        pdf_bytes = generate_pdf_report(
            student_name=student_name, student_id=student_id, date_str=str(lab_date), trials_df=trials_df,
            quiz_score=st.session_state.get("quiz_score", 0), quiz_total=len(QUIZ_QUESTIONS), student_notes=notes)
        st.download_button("Download lab report (.pdf)", data=pdf_bytes, file_name="lab_report.pdf",
                           mime="application/pdf", key="stream_pdf_btn", type="primary",
                           icon=":material/download:")


def render_certificate_section():
    page_header("Certificate", "A certificate of completion, available once you've run the simulation and "
                               "submitted the quiz.")

    has_trial = bool(st.session_state.get("trials"))
    has_quiz = bool(st.session_state.get("quiz_submitted", False))
    quiz_score = st.session_state.get("quiz_score", 0)
    quiz_total = len(QUIZ_QUESTIONS)

    def req(ok, title, sub):
        icon = ICON_CHECK if ok else ICON_LOCK
        tag = "Complete" if ok else "Pending"
        return (f'<div class="vl-req {"ok" if ok else "no"}"><div class="ic">{icon}</div>'
                f'<div><b>{esc(title)}</b><span>{esc(sub)}</span></div><div class="tag">{tag}</div></div>')

    with card("card_requirements"):
        card_title("Requirements")
        ui(
            req(has_trial, "Complete the simulation",
                f"{len(st.session_state.get('trials', []))} trial(s) recorded" if has_trial
                else "Run a query and click “Record current trial”")
            + req(has_quiz, "Submit the quiz",
                  f"Score {quiz_score} / {quiz_total}" if has_quiz else f"Answer all {quiz_total} questions and submit")
        )
        if not (has_trial and has_quiz):
            st.warning("Please complete the virtual lab and give the quiz first — the certificate unlocks once "
                       "both requirements above are complete.", icon=":material/lock:")
            c1, c2, _sp = st.columns([1, 1, 2])
            with c1:
                if not has_trial:
                    st.button("Go to Simulation", icon=":material/science:", on_click=goto, args=("simulation",),
                              width="stretch", type="primary")
            with c2:
                if not has_quiz:
                    st.button("Go to Quiz", icon=":material/quiz:", on_click=goto, args=("quiz",), width="stretch",
                              type="secondary" if not has_trial else "primary")
            return

    perc = int(quiz_score / quiz_total * 100) if quiz_total else 0
    with card("card_cert_name"):
        card_title("Your full name", "It will appear on the certificate exactly as entered.")
        c1, c2 = st.columns([3, 1.2], vertical_alignment="bottom")
        with c1:
            full_name = st.text_input("Full name", key="cert_name", placeholder="e.g. Pankaj Gupta",
                                      label_visibility="collapsed")
        with c2:
            if st.button("Generate certificate", type="primary", icon=":material/workspace_premium:",
                         width="stretch", disabled=not (full_name or "").strip()):
                st.session_state["certificate_generated"] = True
        if not (full_name or "").strip():
            st.caption("Enter your full name to enable certificate generation.")

    full_name = (full_name or "").strip()
    if st.session_state.get("certificate_generated") and full_name:
        cert_date = datetime.now().strftime("%B %d, %Y")
        ui(
            f'<div class="vl-cert"><div class="org">KGIRS VIRTUAL LABORATORY</div>'
            f'<h2>Certificate of Completion</h2><div class="rule"></div>'
            f'<div class="pre">This is to certify that</div><div class="name">{esc(full_name)}</div><div class="line"></div>'
            f'<div class="body">has successfully completed the Virtual Laboratory experiment on '
            f'<b>“{esc(EXPERIMENT_CONFIG["title"])}”</b>, performing the interactive simulation and '
            f'completing the concept assessment quiz.</div>'
            f'<div class="score">Quiz score {quiz_score} / {quiz_total} &middot; {perc}%</div>'
            f'<div class="date">Date of completion: {esc(cert_date)}</div></div>'
        )
        st.write("")
        st.download_button(
            "Download certificate (.pdf)",
            data=generate_certificate_pdf(full_name, quiz_score, quiz_total, cert_date),
            file_name=f"certificate_{re.sub(r'[^A-Za-z0-9]+', '_', full_name).strip('_')}.pdf",
            mime="application/pdf", key="certificate_pdf_btn", type="primary", icon=":material/download:")


# ======================================================================================
# 8. MAIN ENTRYPOINT
# ======================================================================================

def init_session_state():
    defaults = {
        "trials": [], "quiz_answers": {}, "quiz_submitted": False, "quiz_score": 0,
        "certificate_generated": False,
        "sim_source": SOURCE_OPTIONS[0], "q_builtin": DEFAULT_QUERY, "q_paste": "", "q_upload": "",
        "paste_text": "", "k1": SIMULATION_CONFIG["k1_default"], "b": SIMULATION_CONFIG["b_default"],
        "rep_name": "", "rep_id": "", "rep_date": datetime.now().date(),
        "rep_notes": ("The BM25 and TF-IDF rankings were compared across several queries and parameter settings, "
                      "with BM25 showing greater robustness to document length and repeated-term keyword stuffing."),
        "cert_name": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    st.session_state.setdefault("student_info", {"name": "", "id": "", "date": str(datetime.now().date())})


def main():
    st.set_page_config(page_title="BM25 Document Ranking · KGIRS Virtual Lab",
                       page_icon=":material/manage_search:", layout="wide", initial_sidebar_state="auto")

    # Keep widget values alive while their section isn't rendered (Streamlit drops unrendered widget state).
    for k in PERSIST_KEYS:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]
    init_session_state()
    if "nav" not in st.session_state:
        qp = st.query_params.get("section")
        st.session_state["nav"] = qp if qp in NAV_KEYS else "purpose"

    inject_css()
    section, progress_slot = render_sidebar()
    if st.query_params.get("section") != section:
        st.query_params["section"] = section
    scroll_to_top(section)

    {
        "purpose": render_purpose_section,
        "theory": render_theory_section,
        "simulation": render_simulation_section,
        "quiz": render_quiz_section,
        "report": render_report_section,
        "certificate": render_certificate_section,
        "references": render_references_section,
    }[section]()
    render_progress(progress_slot)


if __name__ == "__main__":
    main()
