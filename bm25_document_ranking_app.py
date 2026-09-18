"""
Virtual Laboratory Experiment: BM25 Based Document Ranking (Streamlit)
Implements the BM25 probabilistic ranking function from first principles and
compares its ranking behaviour against the classic TF-IDF vector-space model
(cosine similarity) on a shared document corpus and query.

Structure follows the standard 4-section virtual-lab template:
  1. Theory: Concepts (TF-IDF, BM25), objectives, procedure, key terms, references.
  2. Simulation: Interactive corpus/query/parameter controls, dual ranking engine,
     comparison plots, term-contribution breakdown, and trial logger.
  3. Quiz: Self-grading conceptual assessment with instant feedback.
  4. Report Generation: Student info, recorded trials, observations, and downloadable PDF report.

Note: No custom CSS is used so that Streamlit native light and dark themes render seamlessly,
matching IIT Kharagpur Virtual Labs conventions (Aim / Theory / Procedure / Simulation / Quiz / References).
"""

import os
import re
import math
import collections
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF


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
    "background": """
### Overview & Objective
Given a collection of documents and a user query, a ranking function assigns each document a
relevance score so the most relevant documents can be shown first. This experiment implements
two such functions — **TF-IDF with cosine similarity** and **BM25** — and lets you compare how
they rank the same documents for the same query.

### TF-IDF Vector Space Model
Each document (and the query) is represented as a vector over the vocabulary. Every term's weight
combines **Term Frequency (TF)** — how often the term occurs in that document — with
**Inverse Document Frequency (IDF)** — how rare the term is across the whole collection:

> weight(t, d) = tf(t, d) × idf(t),  where idf(t) = log(N / df(t)) + 1

Relevance is then measured as the **cosine similarity** between the query vector and each document
vector — the cosine of the angle between them, independent of document length. A key limitation of
the classic (raw-count) TF-IDF formulation used here is that term frequency has **no saturation**:
repeating a query term many more times keeps increasing a document's score with no diminishing returns.

### BM25 Ranking Function
BM25 ("Best Matching 25") is a probabilistic ranking function that scores a document D against a
query Q as:

> score(D, Q) = Σ IDF(qᵢ) · [ f(qᵢ, D) · (k1 + 1) ] / [ f(qᵢ, D) + k1 · (1 − b + b · |D| / avgdl) ]

where f(qᵢ, D) is the frequency of query term qᵢ in D, |D| is the document's length in tokens, and
avgdl is the average document length across the collection. Two tunable parameters shape the score:

- **k1** (typically 1.2 – 2.0) controls how quickly additional occurrences of a term **saturate** —
  beyond a point, repeating a term contributes progressively less to the score.
- **b** (0 to 1) controls **document-length normalization** — how much a document's score is
  penalized for being longer than the collection average (b = 1: full normalization, b = 0: none).

### Why BM25 Often Ranks Better than Plain TF-IDF
Because BM25 saturates term-frequency contribution and explicitly normalizes for document length, a
long document that repeats a query term many times (e.g. through keyword stuffing) will not
automatically outrank a shorter, genuinely on-topic document — something plain TF-IDF with raw term
counts is prone to.
    """,
    "procedure": [
        "Step 1: Review the theoretical background, objectives, and key terminology below.",
        "Step 2: Navigate to the Simulation section in the sidebar menu.",
        "Step 3: Choose the built-in sample document corpus, or switch to Custom Documents and paste your own.",
        "Step 4: Enter a search query (a default query is pre-filled for the built-in corpus).",
        "Step 5: Adjust the BM25 parameters k1 (term-frequency saturation) and b (length normalization).",
        "Step 6: Inspect the side-by-side TF-IDF and BM25 rankings, the comparison chart, and the rank "
        "correlation between the two methods.",
        "Step 7: Examine the term-contribution breakdown for the top BM25-ranked document.",
        "Step 8: Click 'Record Current Trial' to log the query, parameters, and results into your session table.",
        "Step 9: Repeat for at least 3-4 different queries and/or k1, b combinations.",
        "Step 10: Complete the assessment Quiz to test your conceptual understanding.",
        "Step 11: Open Report Generation, enter your student information, and download your PDF report."
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

REFERENCES_CONTENT = [
    "S. E. Robertson and K. Sparck Jones, 'Relevance Weighting of Search Terms', Journal of the "
    "American Society for Information Science, 1976.",
    "S. E. Robertson and S. Walker, 'Some Simple Effective Approximations to the 2-Poisson Model for "
    "Probabilistic Weighted Retrieval', SIGIR, 1994.",
    "C. D. Manning, P. Raghavan, and H. Schütze, 'Introduction to Information Retrieval', Cambridge "
    "University Press, 2008 (Chapters 6 & 11: Scoring, Term Weighting, and Probabilistic IR)."
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


# ======================================================================================
# 5. SECTION RENDERERS: THEORY, SIMULATION, QUIZ, REPORT
# ======================================================================================

def render_theory_section():
    """Renders Section 1: Aim, Theory, Background, Objectives, Procedure, and References."""
    st.header("Aim")
    st.write(
        "To implement the BM25 probabilistic ranking function and empirically compare its document "
        "ranking behaviour against the classic TF-IDF vector-space model (cosine similarity)."
    )

    st.divider()
    st.header("Theoretical Framework & Background")
    st.markdown(THEORY_CONTENT["background"])

    st.subheader("Learning Objectives")
    for i, obj in enumerate(EXPERIMENT_CONFIG["objectives"]):
        st.write(f"- **Goal {i+1}**: {obj}")

    st.divider()
    st.subheader("TF-IDF vs. BM25 at a Glance")
    comparison_df = pd.DataFrame(COMPARISON_TABLE, columns=["Aspect", "TF-IDF (cosine similarity)", "BM25"])
    st.table(comparison_df)

    st.divider()
    st.subheader("Experimental Procedure")
    for step in THEORY_CONTENT["procedure"]:
        st.write(f"- {step}")

    st.divider()
    with st.expander("Key Terminology & Variable Reference"):
        var_df = pd.DataFrame(
            list(THEORY_CONTENT["key_terms"].items()),
            columns=["Term / Variable", "Definition & Role"]
        )
        st.table(var_df)

    with st.expander("References"):
        for ref in REFERENCES_CONTENT:
            st.write(f"- {ref}")


def render_simulation_section():
    """Renders Section 2: Interactive corpus/query/parameter controls and dual ranking engine."""
    st.header("Interactive Simulation Sandbox")
    st.info(
        "Choose a document corpus, enter a query, tune the BM25 parameters, and compare the TF-IDF and "
        "BM25 rankings side by side."
    )

    corpus_mode = st.radio(
        "Document Source",
        options=["Built-in Sample Corpus", "Custom Documents"],
        horizontal=True
    )

    if corpus_mode == "Built-in Sample Corpus":
        documents = BUILTIN_CORPUS
        with st.expander("View the built-in corpus documents"):
            for doc in documents:
                st.write(f"**{doc['id']}. {doc['title']}** — {doc['text']}")
        default_query = DEFAULT_QUERY
    else:
        st.caption(
            "Paste your own documents below, separated by a **blank line** between each document "
            "(at least 2 documents are required)."
        )
        custom_text = st.text_area(
            "Custom Documents",
            height=200,
            placeholder="Document one text goes here...\n\nDocument two text goes here...\n\n"
                        "Document three text goes here..."
        )
        documents = parse_custom_documents(custom_text)
        default_query = ""

    query = st.text_input("Search Query", value=default_query)

    col_k1, col_b = st.columns(2)
    cfg = SIMULATION_CONFIG
    with col_k1:
        k1 = st.slider(
            "k1 (Term-Frequency Saturation)",
            min_value=cfg["k1_min"], max_value=cfg["k1_max"],
            value=cfg["k1_default"], step=cfg["k1_step"]
        )
    with col_b:
        b = st.slider(
            "b (Document-Length Normalization)",
            min_value=cfg["b_min"], max_value=cfg["b_max"],
            value=cfg["b_default"], step=cfg["b_step"]
        )

    if len(documents) < 2:
        st.warning("Please provide at least 2 documents (built-in corpus, or paste custom documents above).")
        return
    if not query.strip():
        st.warning("Please enter a search query to run the ranking.")
        return

    ranking_output = rank_documents(documents, query, k1, b)
    results = ranking_output["results"]

    tfidf_sorted = sorted(results, key=lambda r: r["tfidf_score"], reverse=True)
    bm25_sorted = sorted(results, key=lambda r: r["bm25_score"], reverse=True)

    tfidf_order = [r["id"] for r in tfidf_sorted]
    bm25_order = [r["id"] for r in bm25_sorted]
    correlation = spearman_rank_correlation(tfidf_order, bm25_order)

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Top TF-IDF Document", tfidf_sorted[0]["title"][:22] if tfidf_sorted else "N/A")
    with m2:
        st.metric("Top BM25 Document", bm25_sorted[0]["title"][:22] if bm25_sorted else "N/A")
    with m3:
        st.metric("avgdl (tokens)", ranking_output["avgdl"])
    with m4:
        st.metric("Rank Correlation (ρ)", correlation)

    st.subheader("Ranked Results: TF-IDF vs. BM25")
    col_t, col_bm = st.columns(2)
    with col_t:
        st.caption("TF-IDF (cosine similarity) ranking")
        tfidf_df = pd.DataFrame([
            {"Rank": i + 1, "Document": r["title"], "Score": r["tfidf_score"], "Length": r["length"]}
            for i, r in enumerate(tfidf_sorted)
        ])
        st.dataframe(tfidf_df, use_container_width=True, hide_index=True)
    with col_bm:
        st.caption("BM25 ranking")
        bm25_df = pd.DataFrame([
            {"Rank": i + 1, "Document": r["title"], "Score": r["bm25_score"], "Length": r["length"]}
            for i, r in enumerate(bm25_sorted)
        ])
        st.dataframe(bm25_df, use_container_width=True, hide_index=True)

    st.subheader("Score Comparison Chart")
    top_n = min(10, len(results))
    combined_sorted = sorted(results, key=lambda r: (r["tfidf_score"] + r["bm25_score"]), reverse=True)[:top_n]
    titles = [r["title"] for r in combined_sorted]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=titles, y=[r["tfidf_score"] for r in combined_sorted], name="TF-IDF Score"))
    fig.add_trace(go.Bar(x=titles, y=[r["bm25_score"] for r in combined_sorted], name="BM25 Score"))
    fig.update_layout(
        barmode="group",
        title="TF-IDF vs. BM25 Scores by Document",
        xaxis_title="Document",
        yaxis_title="Score",
        height=420,
        margin=dict(l=20, r=20, t=40, b=100),
        xaxis_tickangle=-30
    )
    st.plotly_chart(fig, use_container_width=True)

    if ranking_output["unique_query_terms"] and bm25_sorted:
        st.subheader(f"Term Contribution Breakdown (Top BM25 Document: {bm25_sorted[0]['title']})")
        contributions = bm25_sorted[0]["term_contributions"]
        if contributions:
            contrib_fig = go.Figure()
            contrib_fig.add_trace(go.Bar(
                x=list(contributions.keys()),
                y=list(contributions.values()),
                marker_color="#2563eb"
            ))
            contrib_fig.update_layout(
                title="Per-Term BM25 Score Contribution",
                xaxis_title="Query Term",
                yaxis_title="Contribution to BM25 Score",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(contrib_fig, use_container_width=True)
        else:
            st.info("None of the query terms appear in the top-ranked BM25 document.")

    st.divider()
    st.subheader("Experimental Data Log Book")
    col_log1, col_log2 = st.columns([1.5, 3.5])

    with col_log1:
        st.caption("Capture the current query, parameters, and results into your session trial table:")
        if st.button("Record Current Trial", type="primary", use_container_width=True):
            trial_record = {
                "Trial #": len(st.session_state["trials"]) + 1,
                "Query": query,
                "k1": k1,
                "b": b,
                "Top TF-IDF Doc": tfidf_sorted[0]["title"] if tfidf_sorted else "N/A",
                "Top BM25 Doc": bm25_sorted[0]["title"] if bm25_sorted else "N/A",
                "Rank Correlation": correlation,
                "Timestamp": datetime.now().strftime("%H:%M:%S")
            }
            st.session_state["trials"].append(trial_record)
            st.toast(f"Trial #{trial_record['Trial #']} successfully saved!")

        if st.button("Clear Logged Trials", use_container_width=True):
            st.session_state["trials"] = []
            st.toast("Trial log cleared.")

    with col_log2:
        if st.session_state["trials"]:
            df_trials = pd.DataFrame(st.session_state["trials"])
            st.dataframe(df_trials, use_container_width=True, hide_index=True)
            csv_data = df_trials.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Trials as CSV",
                data=csv_data,
                file_name="bm25_tfidf_trials.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No trials recorded yet. Click 'Record Current Trial' to begin collecting experimental data.")


def render_quiz_section():
    """Renders Section 3: Assessment Quiz with Self-Grading and Feedback."""
    st.header("Concept Assessment Quiz")
    st.write("Answer the conceptual questions below to evaluate your understanding of BM25 and TF-IDF.")

    with st.form("lab_quiz_form"):
        user_responses = {}
        for q in QUIZ_QUESTIONS:
            st.subheader(f"Question {q['id']}")
            st.write(q["question"])
            selected = st.radio(
                label=f"Options for Question {q['id']}:",
                options=q["options"],
                index=st.session_state["quiz_answers"].get(q["id"], 0),
                key=f"quiz_radio_{q['id']}",
                label_visibility="collapsed"
            )
            user_responses[q["id"]] = q["options"].index(selected)

        submitted = st.form_submit_button("Submit Quiz for Grading", type="primary")

    if submitted:
        score = 0
        st.session_state["quiz_answers"] = user_responses
        st.session_state["quiz_submitted"] = True

        st.divider()
        st.subheader("Evaluation Results and Feedback")
        for q in QUIZ_QUESTIONS:
            user_ans = user_responses.get(q["id"])
            correct_ans = q["answer_index"]
            if user_ans == correct_ans:
                score += 1
                st.success(f"**Question {q['id']}: Correct!**\n\n_{q['explanation']}_")
            else:
                st.error(f"**Question {q['id']}: Incorrect.** (Your answer: {q['options'][user_ans]})\n\n"
                         f"**Correct Answer:** {q['options'][correct_ans]}\n\n"
                         f"**Reasoning:** _{q['explanation']}_")

        st.session_state["quiz_score"] = score
        perc = (score / len(QUIZ_QUESTIONS)) * 100
        st.info(f"Final Score: **{score} / {len(QUIZ_QUESTIONS)}** ({perc:.0f}%)")

    elif st.session_state.get("quiz_submitted", False):
        st.success(f"Quiz already submitted. Current score: **{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}**")


def render_report_section():
    """Renders Section 4: Dynamic Lab Report Generator with Guaranteed PDF Export."""
    st.header("Report Generation")
    st.write("Compile your student details, recorded trials, and quiz evaluation into an official PDF report.")

    col1, col2, col3 = st.columns(3)
    with col1:
        student_name = st.text_input("Student Name", value=st.session_state["student_info"].get("name", "Student Name"))
    with col2:
        student_id = st.text_input("Student Roll / ID", value=st.session_state["student_info"].get("id", "EXP-001"))
    with col3:
        lab_date = st.date_input("Experiment Date", value=datetime.now())

    st.session_state["student_info"]["name"] = student_name
    st.session_state["student_info"]["id"] = student_id
    st.session_state["student_info"]["date"] = str(lab_date)

    st.subheader("Discussion & Observations")
    student_notes = st.text_area(
        "Enter your interpretation of results, observations, and conclusions:",
        value=st.session_state.get("student_notes", (
            "The BM25 and TF-IDF rankings were compared across several queries and parameter settings, "
            "with BM25 showing greater robustness to document length and repeated-term keyword stuffing."
        )),
        height=120
    )
    st.session_state["student_notes"] = student_notes

    trials_df = pd.DataFrame(st.session_state["trials"]) if st.session_state["trials"] else pd.DataFrame()

    st.divider()
    st.subheader("Report Summary Preview")
    st.write(f"**Experiment:** {EXPERIMENT_CONFIG['title']}")
    st.write(f"**Student:** {student_name} | **ID:** {student_id} | **Date:** {lab_date}")
    st.write(f"**Quiz Score:** {st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}")

    if not trials_df.empty:
        st.dataframe(trials_df, hide_index=True, use_container_width=True)
    else:
        st.info("Note: You have not recorded any trials in the Simulation tab yet. Your report will indicate 0 trials.")

    pdf_bytes = generate_pdf_report(
        student_name=student_name,
        student_id=student_id,
        date_str=str(lab_date),
        trials_df=trials_df,
        quiz_score=st.session_state.get("quiz_score", 0),
        quiz_total=len(QUIZ_QUESTIONS),
        student_notes=student_notes
    )

    os.makedirs("static", exist_ok=True)
    with open("static/lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    with open("lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)

    st.divider()
    st.subheader("Download Official Lab Report (.pdf)")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.link_button(
            "Open / Download PDF Document",
            url="/app/static/lab_report.pdf",
            type="primary",
            use_container_width=True
        )

    with col_btn2:
        st.download_button(
            label="Download lab_report.pdf",
            data=pdf_bytes,
            file_name="lab_report.pdf",
            mime="application/pdf",
            key="stream_pdf_btn",
            use_container_width=True
        )


# ======================================================================================
# 6. MAIN ENTRYPOINT & NAVIGATION
# ======================================================================================

def init_session_state():
    """Initializes Streamlit session state variables."""
    if "trials" not in st.session_state:
        st.session_state["trials"] = []
    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {
            "name": "Student Name",
            "id": "EXP-001",
            "date": str(datetime.now().date())
        }
    if "student_notes" not in st.session_state:
        st.session_state["student_notes"] = ""


def main():
    st.set_page_config(
        page_title="BM25 Based Document Ranking - Virtual Lab",
        page_icon=None,
        layout="wide"
    )

    init_session_state()

    st.title(EXPERIMENT_CONFIG["title"])

    section = st.sidebar.radio(
        "Lab Navigator",
        options=["Theory", "Simulation", "Quiz", "Report Generation"]
    )

    st.sidebar.divider()
    st.sidebar.subheader("Progress Tracker")
    quiz_status = "Done" if st.session_state.get("quiz_submitted", False) else "Pending"
    st.sidebar.write(f"- **Quiz Status:** {quiz_status}")
    if st.session_state.get("quiz_submitted", False):
        st.sidebar.write(f"- **Quiz Score:** `{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}`")
    st.sidebar.write(f"- **Trials Recorded:** {len(st.session_state.get('trials', []))}")

    if section == "Theory":
        render_theory_section()
    elif section == "Simulation":
        render_simulation_section()
    elif section == "Quiz":
        render_quiz_section()
    elif section == "Report Generation":
        render_report_section()


if __name__ == "__main__":
    main()
