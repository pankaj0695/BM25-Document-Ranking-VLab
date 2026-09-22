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

Note: No custom CSS is used so that Streamlit native light and dark themes render seamlessly.
"""

import io
import re
import math
import random
import collections
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF

def load_custom_css():
    st.markdown(
        """
        <style>

        /* =========================================================
           THEME-AWARE GLOBAL STYLING
           ========================================================= */

        :root {
            --app-bg: var(--background-color);
            --app-text: var(--text-color);
            --app-secondary-text: var(--secondary-text-color);
            --app-border: var(--secondary-background-color);
            --app-card: var(--secondary-background-color);
        }


        /* =========================================================
           MAIN APPLICATION
           ========================================================= */

        .main .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .main {
            background: var(--app-bg);
        }


        /* =========================================================
           MAIN TEXT
           ========================================================= */

        .main h1,
        .main h2,
        .main h3,
        .main h4,
        .main h5,
        .main h6,
        .main p,
        .main li,
        .main label,
        .main .stMarkdown {
            color: var(--app-text);
        }


        /* =========================================================
           SIDEBAR
           ========================================================= */

        [data-testid="stSidebar"] {
            background: #0f172a;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.5rem;
        }


        /* Sidebar text */

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown {
            color: #f8fafc !important;
        }


        /* =========================================================
           SIDEBAR NAVIGATION
           ========================================================= */

        [data-testid="stSidebar"] [data-testid="stRadio"] {
            margin-top: 1rem;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            display: flex;
            align-items: center;

            width: 100%;

            padding: 11px 14px;
            margin: 5px 0;

            border-radius: 10px;

            background: transparent;

            transition:
                background 0.2s ease,
                transform 0.2s ease;

            cursor: pointer;
        }


        /* Hover */

        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(255, 255, 255, 0.08);
            transform: translateX(3px);
        }


        /* Hide radio circles */

        [data-testid="stSidebar"] [data-testid="stRadio"] input {
            display: none !important;
        }

        [data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child {
            display: none !important;
        }


        /* Navigation text */

        [data-testid="stSidebar"] [data-testid="stRadio"] label p {
            margin: 0 !important;
            color: #e5e7eb !important;
            font-size: 0.92rem;
            font-weight: 500;
        }


        /* Active navigation */

        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(
            input:checked
        ) {
            background: rgba(59, 130, 246, 0.18);
        }


        /* =========================================================
           SIDEBAR DIVIDER
           ========================================================= */

        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.15);
        }


        /* =========================================================
           BUTTONS
           ========================================================= */

        .stButton > button,
        .stFormSubmitButton > button {
            border-radius: 10px;
            min-height: 42px;
            font-weight: 650;
        }


        /* =========================================================
           QUIZ
           ========================================================= */

        .quiz-question-number {
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;

            color: #60a5fa;

            font-size: 0.8rem;
            font-weight: 750;

            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .quiz-question-text {
            margin-bottom: 0.8rem;

            color: var(--app-text);

            font-size: 1.08rem;
            font-weight: 650;

            line-height: 1.55;
        }


        /* =========================================================
           METRIC CARDS
           ========================================================= */

        [data-testid="stMetric"] {
            background: var(--app-card);

            border: 1px solid var(--app-border);
            border-radius: 12px;

            padding: 1rem;
        }


        /* =========================================================
           EXPANDERS
           ========================================================= */

        [data-testid="stExpander"] {
            border: 1px solid var(--app-border);
            border-radius: 12px;
        }


        /* =========================================================
           DATAFRAME
           ========================================================= */

        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }


        /* =========================================================
           ALERTS
           ========================================================= */

        [data-testid="stAlert"] {
            border-radius: 10px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )
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

QUIZ_DISPLAY_COUNT = 10
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
    },
        {
        "id": 11,
        "question": "What does TF (Term Frequency) represent in a document?",
        "options": [
            "A) The number of documents containing the term",
            "B) The number of times the term occurs in a particular document",
            "C) The total number of terms in the corpus",
            "D) The average length of all documents"
        ],
        "answer_index": 1,
        "explanation": "Term Frequency represents how many times a particular term occurs within a specific document."
    },

    {
        "id": 12,
        "question": "What does DF (Document Frequency) represent?",
        "options": [
            "A) The number of times a term appears in one document",
            "B) The number of documents in the collection containing the term",
            "C) The total number of tokens in the corpus",
            "D) The number of query terms"
        ],
        "answer_index": 1,
        "explanation": "Document Frequency is the number of documents in the collection that contain a particular term."
    },

    {
        "id": 13,
        "question": "In the TF-IDF formula used in this experiment, what happens to IDF when a term occurs in many documents?",
        "options": [
            "A) IDF generally decreases",
            "B) IDF becomes infinitely large",
            "C) IDF always becomes zero",
            "D) IDF increases linearly with document frequency"
        ],
        "answer_index": 0,
        "explanation": "A term appearing in many documents is less discriminative, so its IDF value becomes smaller."
    },

    {
        "id": 14,
        "question": "What does a high IDF value generally indicate about a term?",
        "options": [
            "A) The term is extremely common",
            "B) The term appears in nearly every document",
            "C) The term is relatively rare across the document collection",
            "D) The term has been removed as a stopword"
        ],
        "answer_index": 2,
        "explanation": "Rare terms provide more distinguishing information and therefore receive higher IDF values."
    },

    {
        "id": 15,
        "question": "Why is cosine similarity used in the TF-IDF implementation?",
        "options": [
            "A) To compare the angle between document and query vectors",
            "B) To count only the number of documents",
            "C) To calculate document length",
            "D) To remove all query terms"
        ],
        "answer_index": 0,
        "explanation": "Cosine similarity measures the similarity in direction between the query vector and document vector."
    },

    {
        "id": 16,
        "question": "What is the main effect of cosine normalization in the vector-space model?",
        "options": [
            "A) It makes all documents contain the same words",
            "B) It reduces the direct effect of vector magnitude",
            "C) It removes IDF from the calculation",
            "D) It makes every similarity score equal to zero"
        ],
        "answer_index": 1,
        "explanation": "Cosine similarity divides the dot product by the magnitudes of the vectors, reducing the direct effect of vector length."
    },

    {
        "id": 17,
        "question": "What does BM25 stand for in this experiment?",
        "options": [
            "A) Binary Matching 25",
            "B) Best Matching 25",
            "C) Boolean Matching 25",
            "D) Basic Model 25"
        ],
        "answer_index": 1,
        "explanation": "BM25 is commonly referred to as Best Matching 25, a probabilistic information-retrieval ranking function."
    },

    {
        "id": 18,
        "question": "Which two parameters are specifically tunable in the BM25 implementation used here?",
        "options": [
            "A) k1 and b",
            "B) TF and IDF",
            "C) DF and TF",
            "D) avgdl and vocabulary size"
        ],
        "answer_index": 0,
        "explanation": "The implementation exposes k1 for term-frequency saturation and b for document-length normalization."
    },

    {
        "id": 19,
        "question": "What happens to the contribution of repeated terms in BM25 as their frequency becomes very large?",
        "options": [
            "A) It increases without any limitation",
            "B) It eventually shows diminishing returns",
            "C) It immediately becomes zero",
            "D) It becomes independent of term frequency"
        ],
        "answer_index": 1,
        "explanation": "BM25 applies term-frequency saturation, meaning additional occurrences contribute progressively less."
    },

    {
        "id": 20,
        "question": "What does a larger k1 generally allow in BM25?",
        "options": [
            "A) Additional occurrences of a term to continue contributing for longer before saturation",
            "B) Document length to become irrelevant",
            "C) IDF to be removed",
            "D) All documents to receive the same score"
        ],
        "answer_index": 0,
        "explanation": "A larger k1 makes the saturation curve less aggressive, allowing repeated occurrences to have more influence before diminishing returns dominate."
    },

    {
        "id": 21,
        "question": "What is the role of b in the BM25 formula?",
        "options": [
            "A) It controls document-length normalization",
            "B) It controls the number of documents",
            "C) It controls tokenization",
            "D) It controls the vocabulary size"
        ],
        "answer_index": 0,
        "explanation": "The parameter b controls how strongly document length relative to avgdl affects the BM25 score."
    },

    {
        "id": 22,
        "question": "What happens when BM25 uses b = 0?",
        "options": [
            "A) Document-length normalization is disabled",
            "B) Term frequency is disabled",
            "C) IDF is disabled",
            "D) Every document receives a score of zero"
        ],
        "answer_index": 0,
        "explanation": "With b = 0, the document-length normalization component no longer changes the score based on document length."
    },

    {
        "id": 23,
        "question": "What happens when b approaches 1?",
        "options": [
            "A) Length normalization becomes stronger",
            "B) Term frequency becomes zero",
            "C) IDF disappears",
            "D) All documents become the same length"
        ],
        "answer_index": 0,
        "explanation": "A larger b gives greater importance to document-length normalization relative to the average document length."
    },

    {
        "id": 24,
        "question": "What does avgdl represent in BM25?",
        "options": [
            "A) Average query length",
            "B) Average document length in the collection",
            "C) Average number of unique terms in the query",
            "D) Average IDF value"
        ],
        "answer_index": 1,
        "explanation": "avgdl is the average number of tokens across the documents in the corpus."
    },

    {
        "id": 25,
        "question": "Why does BM25 use the ratio |D| / avgdl?",
        "options": [
            "A) To compare a document's length with the average document length",
            "B) To calculate the number of query terms",
            "C) To remove stopwords",
            "D) To calculate cosine similarity"
        ],
        "answer_index": 0,
        "explanation": "The ratio tells BM25 whether a document is longer or shorter than the average document in the collection."
    },

    {
        "id": 26,
        "question": "If a document is much longer than avgdl, what effect can BM25's length normalization have?",
        "options": [
            "A) It can reduce the document's score contribution",
            "B) It always increases the score",
            "C) It removes every query term",
            "D) It makes the document impossible to rank"
        ],
        "answer_index": 0,
        "explanation": "When b is positive, documents substantially longer than avgdl receive a stronger normalization effect."
    },

    {
        "id": 27,
        "question": "Why can BM25 reduce the effect of keyword stuffing compared with raw TF-IDF?",
        "options": [
            "A) BM25 ignores all repeated terms",
            "B) BM25 combines term-frequency saturation with document-length normalization",
            "C) BM25 does not use term frequency",
            "D) BM25 only considers document titles"
        ],
        "answer_index": 1,
        "explanation": "BM25 limits the benefit of excessive term repetition through saturation and also accounts for document length."
    },

    {
        "id": 28,
        "question": "In the experiment, what is the purpose of comparing TF-IDF and BM25 on the same corpus and query?",
        "options": [
            "A) To compare how the two ranking methods order the same documents",
            "B) To train a neural network",
            "C) To remove all documents from the corpus",
            "D) To calculate only document length"
        ],
        "answer_index": 0,
        "explanation": "Using the same corpus and query allows the ranking behaviour of TF-IDF and BM25 to be compared directly."
    },

    {
        "id": 29,
        "question": "What does a ranking function primarily do in information retrieval?",
        "options": [
            "A) Assign relevance scores and order documents",
            "B) Permanently delete irrelevant documents",
            "C) Convert every document into an image",
            "D) Train a classification model"
        ],
        "answer_index": 0,
        "explanation": "A ranking function scores documents with respect to a query and orders them according to their estimated relevance."
    },

    {
        "id": 30,
        "question": "What is the purpose of tokenization in this experiment?",
        "options": [
            "A) Split text into individual tokens for processing",
            "B) Calculate Spearman correlation directly",
            "C) Generate PDF certificates",
            "D) Select the value of k1"
        ],
        "answer_index": 0,
        "explanation": "Tokenization converts raw text into individual terms that can be counted and used during ranking."
    },

    {
        "id": 31,
        "question": "What happens to tokens that belong to the built-in STOPWORDS set?",
        "options": [
            "A) They are filtered out before scoring",
            "B) They are given the highest IDF",
            "C) They are repeated five times",
            "D) They become document titles"
        ],
        "answer_index": 0,
        "explanation": "The experiment removes common stopwords before ranking because they generally provide little discriminative information."
    },

    {
        "id": 32,
        "question": "Why are stopwords commonly removed in information retrieval?",
        "options": [
            "A) They are often very common and provide limited discriminative information",
            "B) They are always misspelled",
            "C) They contain only numbers",
            "D) They cannot be represented as strings"
        ],
        "answer_index": 0,
        "explanation": "Very common words tend to occur across many documents and therefore contribute relatively little to distinguishing documents."
    },

    {
        "id": 33,
        "question": "In this experiment, what happens if a query term does not occur in any document?",
        "options": [
            "A) Its TF-IDF IDF contribution is treated as zero",
            "B) Every document receives an infinite score",
            "C) The application crashes automatically",
            "D) The term becomes a stopword permanently"
        ],
        "answer_index": 0,
        "explanation": "The implementation returns an IDF value of 0 for a term with document frequency zero in the TF-IDF calculation."
    },

    {
        "id": 34,
        "question": "What is document frequency (DF) used for when calculating IDF?",
        "options": [
            "A) To determine how widely a term is distributed across documents",
            "B) To determine the number of query results displayed",
            "C) To determine the value of b",
            "D) To determine the average document title length"
        ],
        "answer_index": 0,
        "explanation": "DF counts how many documents contain a term and is used to determine its inverse document frequency."
    },

    {
        "id": 35,
        "question": "Suppose two documents contain the same query term with the same frequency, but one document is much longer. With positive b, what can happen in BM25?",
        "options": [
            "A) The longer document can receive a stronger length-normalization penalty",
            "B) The longer document is always ranked first",
            "C) Document length is completely ignored",
            "D) Both documents must receive identical BM25 scores"
        ],
        "answer_index": 0,
        "explanation": "BM25 accounts for document length relative to avgdl, so a substantially longer document can be penalized when b is positive."
    },

    {
        "id": 36,
        "question": "What is the main difference between raw TF-IDF term-frequency handling and BM25 term-frequency handling in this experiment?",
        "options": [
            "A) Raw TF-IDF is unbounded while BM25 applies saturation",
            "B) TF-IDF has saturation while BM25 does not",
            "C) Neither method uses term frequency",
            "D) Both methods completely ignore repeated terms"
        ],
        "answer_index": 0,
        "explanation": "The classic raw TF-IDF implementation used here grows with term frequency, whereas BM25 introduces diminishing returns."
    },

    {
        "id": 37,
        "question": "Which BM25 parameter would you investigate when studying the effect of repeated occurrences of a query term?",
        "options": [
            "A) k1",
            "B) b",
            "C) avgdl only",
            "D) Number of documents"
        ],
        "answer_index": 0,
        "explanation": "k1 controls the shape and strength of term-frequency saturation in BM25."
    },

    {
        "id": 38,
        "question": "Which BM25 parameter would you investigate when studying the effect of document length?",
        "options": [
            "A) k1",
            "B) b",
            "C) TF only",
            "D) Query vocabulary size"
        ],
        "answer_index": 1,
        "explanation": "b controls the degree of document-length normalization."
    },

    {
        "id": 39,
        "question": "What is the purpose of the comparison chart in the simulation?",
        "options": [
            "A) To visually compare document scores or rankings produced by the two methods",
            "B) To train the BM25 model",
            "C) To create new documents",
            "D) To calculate stopwords"
        ],
        "answer_index": 0,
        "explanation": "The visualization helps students inspect how TF-IDF and BM25 score and rank the same documents."
    },

    {
        "id": 40,
        "question": "What does Spearman rank correlation measure in this experiment?",
        "options": [
            "A) How similarly two ranking methods order the same documents",
            "B) The number of tokens in a document",
            "C) The IDF of a query term",
            "D) The value of k1"
        ],
        "answer_index": 0,
        "explanation": "Spearman's rank correlation measures the similarity between two rankings based on the relative ordering of their items."
    },

    {
        "id": 41,
        "question": "If two ranking methods produce exactly the same ordering of documents, what would their Spearman rank correlation be?",
        "options": [
            "A) -1",
            "B) 0",
            "C) 1",
            "D) 100"
        ],
        "answer_index": 2,
        "explanation": "A Spearman correlation of 1 indicates identical ranking order."
    },

    {
        "id": 42,
        "question": "What would a negative Spearman rank correlation generally indicate?",
        "options": [
            "A) The two rankings tend to order documents in opposite directions",
            "B) The two rankings are identical",
            "C) There are no documents",
            "D) Every document has the same length"
        ],
        "answer_index": 0,
        "explanation": "A negative rank correlation indicates that higher positions in one ranking tend to correspond to lower positions in the other."
    },

    {
        "id": 43,
        "question": "Why is it useful to record multiple trials in this experiment?",
        "options": [
            "A) To observe how queries and BM25 parameters affect ranking results",
            "B) To permanently change the BM25 formula",
            "C) To delete previous results",
            "D) To increase the number of stopwords"
        ],
        "answer_index": 0,
        "explanation": "Recording multiple trials allows students to compare ranking behaviour under different queries and parameter settings."
    },

    {
        "id": 44,
        "question": "If k1 is changed while b remains constant, which aspect of BM25 are you primarily investigating?",
        "options": [
            "A) Term-frequency saturation",
            "B) Document tokenization",
            "C) PDF generation",
            "D) Sidebar navigation"
        ],
        "answer_index": 0,
        "explanation": "Changing k1 primarily changes how quickly the contribution from repeated term occurrences saturates."
    },

    {
        "id": 45,
        "question": "If b is changed while k1 remains constant, which aspect of BM25 are you primarily investigating?",
        "options": [
            "A) Document-length normalization",
            "B) Query tokenization",
            "C) Stopword vocabulary",
            "D) Cosine vector dimensions only"
        ],
        "answer_index": 0,
        "explanation": "Changing b changes the strength of document-length normalization."
    },

    {
        "id": 46,
        "question": "Why can TF-IDF and BM25 produce different rankings for the same query?",
        "options": [
            "A) They handle term frequency and document length differently",
            "B) They always use different document collections",
            "C) BM25 does not process text",
            "D) TF-IDF cannot rank documents"
        ],
        "answer_index": 0,
        "explanation": "The methods use different scoring formulations, particularly in how they handle repeated terms and document length."
    },

    {
        "id": 47,
        "question": "What happens to the BM25 score contribution when a query term has frequency zero in a document?",
        "options": [
            "A) That term contributes nothing to the document's BM25 score",
            "B) The term receives the maximum possible score",
            "C) The entire document is deleted",
            "D) The term is counted as occurring once"
        ],
        "answer_index": 0,
        "explanation": "If the query term does not occur in a document, its term frequency is zero and it contributes nothing to that document's BM25 score."
    },

    {
        "id": 48,
        "question": "What is the purpose of examining the term-contribution breakdown for a top BM25-ranked document?",
        "options": [
            "A) To understand which query terms contributed to its BM25 score",
            "B) To change the document's original text",
            "C) To remove all query terms",
            "D) To calculate the PDF size"
        ],
        "answer_index": 0,
        "explanation": "The contribution breakdown helps students understand how individual query terms contribute to the final BM25 score."
    },

    {
        "id": 49,
        "question": "What is a useful experimental approach when studying the effect of BM25 parameters?",
        "options": [
            "A) Change one parameter while keeping other important settings constant and compare rankings",
            "B) Change every parameter randomly at the same time",
            "C) Never compare the rankings",
            "D) Delete the corpus after every trial"
        ],
        "answer_index": 0,
        "explanation": "Changing one parameter at a time makes it easier to understand that parameter's effect on ranking behaviour."
    },

    {
        "id": 50,
        "question": "What is the main learning outcome of comparing BM25 with TF-IDF in this virtual laboratory?",
        "options": [
            "A) Understanding how different ranking formulations respond to term frequency, document length, and query terms",
            "B) Learning how to build a neural network from scratch",
            "C) Learning how to create a database",
            "D) Learning how to compress PDF files"
        ],
        "answer_index": 0,
        "explanation": "The experiment is designed to show how BM25 and TF-IDF rank the same documents differently and how BM25's k1 and b parameters affect those rankings."
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

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(148, 163, 184)
    pdf.set_xy(0, page_h - 20)
    pdf.cell(page_w, 5, EXPERIMENT_CONFIG["title"] + " | KGIRS Virtual Lab", align="C")

    return bytes(pdf.output())


def is_lab_completed() -> bool:
    """A student has completed the lab once they've recorded at least one simulation trial and
    submitted the quiz."""
    has_trial = bool(st.session_state.get("trials"))
    has_quiz = bool(st.session_state.get("quiz_submitted", False))
    return has_trial and has_quiz


# ======================================================================================
# 5. SECTION RENDERERS: PURPOSE, THEORY, SIMULATION, QUIZ, REPORT, CERTIFICATE, REFERENCES
# ======================================================================================

def render_purpose_section():
    """Renders the Purpose section: the objective of this virtual lab experiment."""
    st.header("Purpose")
    st.write(
        "To implement the BM25 probabilistic ranking function and empirically compare its document "
        "ranking behaviour against the classic TF-IDF vector-space model (cosine similarity)."
    )
    st.subheader("Learning Objectives")
    for i, obj in enumerate(EXPERIMENT_CONFIG["objectives"]):
        st.write(f"- **Goal {i+1}**: {obj}")


def render_algorithm_diagram():
    """Draws a flowchart illustrating how a query and corpus flow through the ranking pipeline."""
    fig = go.Figure()

    boxes = [
        {"id": "input", "x": 0.5, "y": 0, "w": 1.7, "h": 0.85,
         "text": "Document Corpus<br>+ Search Query", "color": "#e0f2fe", "border": "#0284c7"},
        {"id": "tok", "x": 2.7, "y": 0, "w": 1.9, "h": 0.85,
         "text": "Tokenization &<br>Stopword Removal", "color": "#e0f2fe", "border": "#0284c7"},
        {"id": "freq", "x": 4.9, "y": 0, "w": 2.0, "h": 0.85,
         "text": "Term / Document<br>Frequency Counting", "color": "#e0f2fe", "border": "#0284c7"},
        {"id": "tfidf", "x": 7.3, "y": 1.05, "w": 2.3, "h": 0.85,
         "text": "TF-IDF Weighting →<br>Cosine Similarity", "color": "#dbeafe", "border": "#2563eb"},
        {"id": "bm25", "x": 7.3, "y": -1.05, "w": 2.3, "h": 0.85,
         "text": "BM25 Scoring<br>(k1 saturation, b length-norm)", "color": "#fef3c7", "border": "#d97706"},
        {"id": "rank", "x": 9.9, "y": 0, "w": 2.0, "h": 0.85,
         "text": "Ranked Document<br>List (per method)", "color": "#dcfce7", "border": "#16a34a"},
    ]

    for b in boxes:
        fig.add_shape(
            type="rect",
            x0=b["x"] - b["w"] / 2, x1=b["x"] + b["w"] / 2,
            y0=b["y"] - b["h"] / 2, y1=b["y"] + b["h"] / 2,
            line=dict(color=b["border"], width=2),
            fillcolor=b["color"]
        )
        fig.add_annotation(
            x=b["x"], y=b["y"], text=b["text"], showarrow=False,
            font=dict(size=12, color="#0f172a"), align="center"
        )

    arrows = [
        ("input", "tok"), ("tok", "freq"),
        ("freq", "tfidf"), ("freq", "bm25"),
        ("tfidf", "rank"), ("bm25", "rank"),
    ]
    box_by_id = {b["id"]: b for b in boxes}
    for src, dst in arrows:
        b1, b2 = box_by_id[src], box_by_id[dst]
        x0 = b1["x"] + b1["w"] / 2
        x1 = b2["x"] - b2["w"] / 2
        y0 = b1["y"] if b1["y"] == b2["y"] else b1["y"] + (0.15 if b2["y"] > b1["y"] else -0.15)
        y1 = b2["y"]
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1.8, arrowcolor="#64748b"
        )

    fig.update_xaxes(visible=False, range=[-0.5, 11.5])
    fig.update_yaxes(visible=False, range=[-1.9, 1.9], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white", showlegend=False
    )
    st.plotly_chart(fig, width="stretch")


def render_theory_section():
    """Renders the Theory section: Background, algorithm diagram, applications, procedure, key terms."""
    st.header("Theoretical Framework & Background")
    st.markdown(THEORY_CONTENT["background"], unsafe_allow_html=True)
    st.caption("Tip: terms underlined with dots (e.g. TF-IDF, BM25, k1, b) show a short definition on hover.")

    st.divider()
    st.subheader("TF-IDF vs. BM25 at a Glance")
    comparison_df = pd.DataFrame(COMPARISON_TABLE, columns=["Aspect", "TF-IDF (cosine similarity)", "BM25"])
    st.table(comparison_df)

    st.divider()
    st.subheader("How the Algorithm Works: Ranking Pipeline")
    st.write(
        "The diagram below traces how a query and document corpus flow through tokenization, "
        "frequency counting, and finally the two parallel scoring paths — TF-IDF cosine similarity "
        "and BM25 — that this lab compares."
    )
    render_algorithm_diagram()

    st.divider()
    st.subheader("Applications of BM25")
    st.write("BM25 is not just a textbook formula — it is the working relevance engine behind many "
              "real-world retrieval systems:")
    for name, desc in BM25_APPLICATIONS:
        st.write(f"- **{name}**: {desc}")

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


def render_references_section():
    """Renders the References section: citations with direct links to the source papers."""
    st.header("References")
    st.write("The theoretical background in this virtual lab is drawn from the following foundational "
              "papers and texts:")
    for ref in REFERENCES_CONTENT:
        st.markdown(f"- {ref['citation']} [↗ Link]({ref['url']})")


def render_simulation_section():
    """Renders Section 2: Interactive corpus/query/parameter controls and dual ranking engine."""
    st.header("Interactive Simulation Sandbox")
    st.info(
        "Choose a document corpus, enter a query, tune the BM25 parameters, and compare the TF-IDF and "
        "BM25 rankings side by side."
    )

    corpus_mode = st.radio(
        "Document Source",
        options=["Built-in Sample Corpus", "Custom Documents", "Upload PDF / DOCX Files"],
        horizontal=True
    )

    if corpus_mode == "Built-in Sample Corpus":
        documents = BUILTIN_CORPUS
        with st.expander("View the built-in corpus documents"):
            for doc in documents:
                st.write(f"**{doc['id']}. {doc['title']}** — {doc['text']}")
        default_query = DEFAULT_QUERY
    elif corpus_mode == "Custom Documents":
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
    else:
        st.caption(
            "Upload two or more PDF or DOCX files. Each file becomes one document; its text is "
            "extracted automatically and used for ranking (requires the `pypdf` and `python-docx` "
            "packages)."
        )
        uploaded_files = st.file_uploader(
            "Upload PDF / DOCX Documents",
            type=["pdf", "docx"],
            accept_multiple_files=True
        )
        documents = []
        if uploaded_files:
            for i, uf in enumerate(uploaded_files):
                extracted = extract_text_from_upload(uf)
                if extracted:
                    documents.append({"id": i + 1, "title": uf.name, "text": extracted})
                else:
                    st.warning(f"Could not extract text from **{uf.name}** — it may be empty, "
                               f"image-only, or an unsupported file.")
            if documents:
                with st.expander("Preview extracted text"):
                    for doc in documents:
                        preview = doc["text"][:400] + ("..." if len(doc["text"]) > 400 else "")
                        st.write(f"**{doc['id']}. {doc['title']}**")
                        st.caption(preview)
        default_query = ""

    query = st.text_input("Search Query", value=default_query)

    col_k1, col_b = st.columns(2)
    cfg = SIMULATION_CONFIG
    with col_k1:
        k1 = st.slider(
            "k1 (Term-Frequency Saturation)",
            min_value=cfg["k1_min"], max_value=cfg["k1_max"],
            value=cfg["k1_default"], step=cfg["k1_step"],
            help=GLOSSARY["k1"]
        )
    with col_b:
        b = st.slider(
            "b (Document-Length Normalization)",
            min_value=cfg["b_min"], max_value=cfg["b_max"],
            value=cfg["b_default"], step=cfg["b_step"],
            help=GLOSSARY["b"]
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
        st.metric("avgdl (tokens)", ranking_output["avgdl"], help=GLOSSARY["avgdl"])
    with m4:
        st.metric("Rank Correlation (ρ)", correlation, help=GLOSSARY["rank correlation"])

    st.divider()
    st.subheader("Behind the Scenes: How the Query is Actually Matched")
    st.caption(
        "A live look at what the ranking engine does internally: tokenizing your query, matching it "
        "against every document, funneling down to the top results, and projecting documents and the "
        "query into a shared 2D space."
    )

    top_k = min(5, len(bm25_sorted))
    retrieved_ids = {r["id"] for r in bm25_sorted[:top_k]}

    viz_tab1, viz_tab2, viz_tab3 = st.tabs([
        "1. Tokenization & Term Matching", "2. Retrieval Funnel", "3. Document/Query Embedding Space"
    ])

    with viz_tab1:
        st.write("**Tokenized query terms** (after lowercasing, stopword removal, and cleanup):")
        query_terms = ranking_output["unique_query_terms"]
        if query_terms:
            chip_html = " ".join(
                f'<span style="background:#2563eb; color:white; padding:3px 10px; border-radius:12px; '
                f'margin:3px; display:inline-block; font-size:0.85em;">{t}</span>'
                for t in query_terms
            )
            st.markdown(chip_html, unsafe_allow_html=True)
        else:
            st.info("No meaningful query terms remain after tokenization/stopword removal.")

        st.write("")
        st.write("**Term-Document Match Heatmap** — which query terms appear in which documents, and "
                  "how many times (raw term frequency):")
        if query_terms and results:
            heat_docs = [r["title"][:28] for r in results]
            heat_z = []
            for term in query_terms:
                row = []
                for doc in documents:
                    toks = tokenize(doc["text"])
                    row.append(collections.Counter(toks).get(term, 0))
                heat_z.append(row)
            heat_fig = go.Figure(data=go.Heatmap(
                z=heat_z, x=heat_docs, y=query_terms,
                colorscale="Blues", showscale=True,
                hovertemplate="Doc: %{x}<br>Term: %{y}<br>Frequency: %{z}<extra></extra>"
            ))
            heat_fig.update_layout(
                title="Query-Term Frequency per Document",
                height=max(260, 60 * len(query_terms) + 120),
                margin=dict(l=20, r=20, t=40, b=100),
                xaxis_tickangle=-30
            )
            st.plotly_chart(heat_fig, width="stretch")

    with viz_tab2:
        candidates = [r for r in results if any(r["term_contributions"].get(t, 0) > 0 for t in query_terms)] \
            if query_terms else []
        funnel_fig = go.Figure(go.Funnel(
            y=["Total Documents in Corpus", "Candidates (≥1 matching term)", f"Top-{top_k} Retrieved"],
            x=[len(documents), len(candidates), top_k],
            textinfo="value+percent initial",
            marker={"color": ["#93c5fd", "#3b82f6", "#1e3a8a"]}
        ))
        funnel_fig.update_layout(
            title="Retrieval Funnel: From Full Corpus to Top-K Results",
            height=360, margin=dict(l=20, r=100, t=40, b=20)
        )
        st.plotly_chart(funnel_fig, width="stretch")
        st.caption(
            f"Out of **{len(documents)}** documents in the corpus, **{len(candidates)}** contain at "
            f"least one query term, and the top **{top_k}** (by BM25 score) are treated as retrieved."
        )

    with viz_tab3:
        st.write(
            "Each document and the query are represented as TF-IDF vectors and projected down to 2D "
            f"(via SVD) so their relative positions can be visualized. Documents in the **top-{top_k} "
            "retrieved** set are highlighted and connected to the query with a dotted line."
        )
        projection = build_embedding_projection(documents, query, retrieved_ids)
        points = projection["points"]
        if points:
            query_pt = next(p for p in points if p["id"] == "query")
            other_pts = [p for p in points if p["id"] != "query" and not p["retrieved"]]
            retrieved_pts = [p for p in points if p["id"] != "query" and p["retrieved"]]

            emb_fig = go.Figure()
            for p in retrieved_pts:
                emb_fig.add_trace(go.Scatter(
                    x=[query_pt["x"], p["x"]], y=[query_pt["y"], p["y"]],
                    mode="lines", line=dict(color="#94a3b8", dash="dot", width=1),
                    showlegend=False, hoverinfo="skip"
                ))
            if other_pts:
                emb_fig.add_trace(go.Scatter(
                    x=[p["x"] for p in other_pts], y=[p["y"] for p in other_pts],
                    mode="markers", name="Other Documents",
                    marker=dict(size=10, color="#cbd5e1", line=dict(width=1, color="#94a3b8")),
                    text=[p["title"] for p in other_pts], hovertemplate="%{text}<extra></extra>"
                ))
            if retrieved_pts:
                emb_fig.add_trace(go.Scatter(
                    x=[p["x"] for p in retrieved_pts], y=[p["y"] for p in retrieved_pts],
                    mode="markers", name=f"Top-{top_k} Retrieved",
                    marker=dict(size=14, color="#2563eb", line=dict(width=1.5, color="#1e3a8a")),
                    text=[p["title"] for p in retrieved_pts], hovertemplate="%{text}<extra></extra>"
                ))
            emb_fig.add_trace(go.Scatter(
                x=[query_pt["x"]], y=[query_pt["y"]], mode="markers+text", name="Query",
                marker=dict(size=18, color="#f97316", symbol="star", line=dict(width=1.5, color="#7c2d12")),
                text=["Query"], textposition="top center", hoverinfo="skip"
            ))
            emb_fig.update_layout(
                title="2D Projection of Document & Query TF-IDF Vectors",
                height=460, margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Component 1", yaxis_title="Component 2"
            )
            st.plotly_chart(emb_fig, width="stretch")
        else:
            st.info("Not enough vocabulary to build an embedding projection yet.")

    st.divider()
    st.subheader("Ranked Results: TF-IDF vs. BM25")
    col_t, col_bm = st.columns(2)
    with col_t:
        st.caption("TF-IDF (cosine similarity) ranking")
        tfidf_df = pd.DataFrame([
            {"Rank": i + 1, "Document": r["title"], "Score": r["tfidf_score"], "Length": r["length"]}
            for i, r in enumerate(tfidf_sorted)
        ])
        st.dataframe(tfidf_df, width="stretch", hide_index=True)
    with col_bm:
        st.caption("BM25 ranking")
        bm25_df = pd.DataFrame([
            {"Rank": i + 1, "Document": r["title"], "Score": r["bm25_score"], "Length": r["length"]}
            for i, r in enumerate(bm25_sorted)
        ])
        st.dataframe(bm25_df, width="stretch", hide_index=True)

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
    st.plotly_chart(fig, width="stretch")

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
            st.plotly_chart(contrib_fig, width="stretch")
        else:
            st.info("None of the query terms appear in the top-ranked BM25 document.")

    st.divider()
    st.subheader("Experimental Data Log Book")
    col_log1, col_log2 = st.columns([1.5, 3.5])

    with col_log1:
        st.caption("Capture the current query, parameters, and results into your session trial table:")
        if st.button("Record Current Trial", type="primary", width="stretch"):
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

        if st.button("Clear Logged Trials", width="stretch"):
            st.session_state["trials"] = []
            st.toast("Trial log cleared.")

    with col_log2:
        if st.session_state["trials"]:
            df_trials = pd.DataFrame(st.session_state["trials"])
            st.dataframe(df_trials, width="stretch", hide_index=True)
            csv_data = df_trials.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Trials as CSV",
                data=csv_data,
                file_name="bm25_tfidf_trials.csv",
                mime="text/csv",
                width="stretch"
            )
        else:
            st.info("No trials recorded yet. Click 'Record Current Trial' to begin collecting experimental data.")


def render_quiz_section():
    """Renders the BM25 assessment quiz with 10 randomized questions."""

    st.header("Concept Assessment Quiz")

    st.write(
        "Answer all 10 questions below to evaluate your understanding "
        "of BM25 and TF-IDF."
    )

    # --------------------------------------------------
    # GET THE SAME 10 QUESTIONS FOR THIS SESSION
    # --------------------------------------------------

    quiz_questions = st.session_state["selected_quiz_questions"]

    # --------------------------------------------------
    # INITIALIZE QUIZ STATE
    # --------------------------------------------------

    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}

    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False

    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0

    user_responses = {}

    # --------------------------------------------------
    # DISPLAY ONLY 10 QUESTIONS
    # --------------------------------------------------

    for question_number, q in enumerate(quiz_questions, start=1):

        st.markdown(
            f'<div class="quiz-question-number">'
            f'QUESTION {question_number}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="quiz-question-text">'
            f'{q["question"]}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Check previous answer
        previous_answer = st.session_state["quiz_answers"].get(q["id"])

        if previous_answer is None:
            default_index = None
        else:
            default_index = previous_answer

        selected = st.radio(
            label=f"Options for Question {question_number}",
            options=q["options"],
            index=default_index,
            key=f"quiz_radio_{question_number}",
            label_visibility="collapsed"
        )

        # Store answer only when selected
        if selected is not None:
            user_responses[q["id"]] = q["options"].index(selected)

        st.write("")

    # --------------------------------------------------
    # ANSWER COUNT
    # --------------------------------------------------

    total_questions = len(quiz_questions)

    answered_questions = len(user_responses)

    all_answered = answered_questions == total_questions

    # --------------------------------------------------
    # STATUS
    # --------------------------------------------------

    if all_answered:

        st.success(
            f"All {total_questions} questions answered. "
            "You can now submit the quiz."
        )

    else:

        remaining = total_questions - answered_questions

        st.info(
            f"Please answer all questions before submitting. "
            f"{answered_questions} / {total_questions} answered "
            f"({remaining} remaining)."
        )

    # --------------------------------------------------
    # SUBMIT BUTTON
    # --------------------------------------------------

    submitted = st.button(
        "Submit Quiz for Grading",
        type="primary",
        width="stretch",
        disabled=not all_answered
    )

    # --------------------------------------------------
    # GRADING
    # --------------------------------------------------

    if submitted:

        st.session_state["quiz_answers"] = user_responses

        st.session_state["quiz_submitted"] = True

        score = 0

        st.divider()

        st.subheader("Evaluation Results and Feedback")

        # IMPORTANT:
        # This loop ONLY grades questions.
        # DO NOT create st.radio() here.

        for question_number, q in enumerate(
            quiz_questions,
            start=1
        ):

            user_ans = user_responses.get(q["id"])

            correct_ans = q["answer_index"]

            if user_ans == correct_ans:

                score += 1

                st.success(
                    f"**Question {question_number}: Correct!**\n\n"
                    f"_{q['explanation']}_"
                )

            else:

                st.error(
                    f"**Question {question_number}: Incorrect.**\n\n"
                    f"Your answer: "
                    f"{q['options'][user_ans]}\n\n"
                    f"**Correct Answer:** "
                    f"{q['options'][correct_ans]}\n\n"
                    f"**Reasoning:** "
                    f"_{q['explanation']}_"
                )

        # Save score
        st.session_state["quiz_score"] = score

        percentage = (score / total_questions) * 100

        st.info(
            f"Final Score: **{score} / {total_questions}** "
            f"({percentage:.0f}%)"
        )
        
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
    st.write(
    f"**Quiz Score:** "
    f"{st.session_state.get('quiz_score', 0)} / {QUIZ_DISPLAY_COUNT}"
)

    if not trials_df.empty:
        st.dataframe(trials_df, hide_index=True, width="stretch")
    else:
        st.info("Note: You have not recorded any trials in the Simulation tab yet. Your report will indicate 0 trials.")

    pdf_bytes = generate_pdf_report(
        student_name=student_name,
        student_id=student_id,
        date_str=str(lab_date),
        trials_df=trials_df,
        quiz_score=st.session_state.get("quiz_score", 0),
        quiz_total=QUIZ_DISPLAY_COUNT,
        student_notes=student_notes
    )

    st.divider()
    st.subheader("Download Official Lab Report (.pdf)")

    st.download_button(
        label="Download lab_report.pdf",
        data=pdf_bytes,
        file_name="lab_report.pdf",
        mime="application/pdf",
        key="stream_pdf_btn",
        type="primary",
        width="stretch"
    )


def render_certificate_section():
    """Renders the Certificate section: eligibility gate, full-name capture, and PDF generation."""
    st.header("Certificate of Completion")
    st.write(
        "Generate a personalized certificate confirming that you completed this virtual lab "
        "experiment and your quiz score."
    )

    has_trial = bool(st.session_state.get("trials"))
    has_quiz = bool(st.session_state.get("quiz_submitted", False))

    if not (has_trial and has_quiz):
        st.warning(
            "You haven't completed the virtual lab yet. Please complete the following before a "
            "certificate can be generated:"
        )
        st.write(f"- {'✅' if has_trial else '❌'} Run at least one trial in the **Simulation** "
                 f"section and click 'Record Current Trial'.")
        st.write(f"- {'✅' if has_quiz else '❌'} Complete and submit the **Quiz**.")
        st.info("Once both steps are done, come back here to generate your certificate.")
        return

    st.success("You have completed the simulation and the quiz — you're eligible for a certificate!")

    quiz_score = st.session_state.get("quiz_score", 0)
    quiz_total = QUIZ_DISPLAY_COUNT
    perc = int((quiz_score / quiz_total) * 100) if quiz_total else 0

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Quiz Score", f"{quiz_score} / {quiz_total}")
    with m2:
        st.metric("Percentage", f"{perc}%")

    st.divider()
    st.subheader("Enter Your Full Name")
    st.caption("Your full name will appear on the certificate exactly as entered below.")
    full_name = st.text_input(
        "Full Name",
        value=st.session_state.get("certificate_full_name", ""),
        placeholder="e.g. Pankaj Gupta"
    )
    st.session_state["certificate_full_name"] = full_name

    generate_clicked = st.button(
        "Generate Certificate", type="primary", width="stretch", disabled=not full_name.strip()
    )
    if not full_name.strip():
        st.caption("Enter your full name above to enable certificate generation.")

    if generate_clicked:
        st.session_state["certificate_generated"] = True

    if st.session_state.get("certificate_generated") and full_name.strip():
        cert_date = datetime.now().strftime("%B %d, %Y")

        st.divider()
        st.subheader("Certificate Preview")
        st.markdown(
            f"""
<div style="border: 3px solid #1e3a8a; border-radius: 6px; padding: 32px; text-align: center;
            background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);">
    <div style="font-size: 0.85em; letter-spacing: 2px; color: #1e3a8a; font-weight: 600;">
        KGIRS VIRTUAL LABORATORY
    </div>
    <div style="font-size: 2em; font-weight: 700; color: #0f172a; margin: 10px 0;">
        Certificate of Completion
    </div>
    <div style="width: 80px; height: 2px; background: #cbd5e1; margin: 12px auto;"></div>
    <div style="font-size: 1em; color: #475569;">This is to certify that</div>
    <div style="font-size: 1.6em; font-weight: 700; color: #1e3a8a; margin: 10px 0;">
        {full_name.strip()}
    </div>
    <div style="width: 160px; height: 2px; background: #1e3a8a; margin: 6px auto 16px;"></div>
    <div style="font-size: 1em; color: #334155; max-width: 640px; margin: 0 auto; line-height: 1.6;">
        has successfully completed the Virtual Laboratory experiment on
        <b>"{EXPERIMENT_CONFIG['title']}"</b>, performing the interactive simulation and completing
        the concept assessment quiz with a final score of <b>{quiz_score} / {quiz_total} ({perc}%)</b>.
    </div>
    <div style="margin-top: 18px; font-size: 0.9em; color: #64748b;">
        Date of Completion: {cert_date}
    </div>
</div>
""",
            unsafe_allow_html=True
        )

        cert_bytes = generate_certificate_pdf(
            student_name=full_name.strip(),
            quiz_score=quiz_score,
            quiz_total=quiz_total,
            date_str=cert_date
        )

        st.divider()
        st.download_button(
            label="Download Certificate (.pdf)",
            data=cert_bytes,
            file_name=f"certificate_{full_name.strip().replace(' ', '_')}.pdf",
            mime="application/pdf",
            key="certificate_pdf_btn",
            type="primary",
            width="stretch"
        )


# ======================================================================================
# 6. MAIN ENTRYPOINT & NAVIGATION
# ======================================================================================

def init_session_state():

    if "trials" not in st.session_state:
        st.session_state["trials"] = []

    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}

    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False

    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0

    # Randomly select 10 questions ONCE per session
    if "selected_quiz_questions" not in st.session_state:
        st.session_state["selected_quiz_questions"] = random.sample(
            QUIZ_QUESTIONS,
            min(10, len(QUIZ_QUESTIONS))
        )

    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {
            "name": "Student Name",
            "id": "EXP-001",
            "date": str(datetime.now().date())
        }

    if "student_notes" not in st.session_state:
        st.session_state["student_notes"] = ""

    if "certificate_full_name" not in st.session_state:
        st.session_state["certificate_full_name"] = ""

    if "certificate_generated" not in st.session_state:
        st.session_state["certificate_generated"] = False
    if "selected_quiz_questions" not in st.session_state:
        st.session_state["selected_quiz_questions"] = random.sample(
        QUIZ_QUESTIONS,
        10
    )


def main():
    st.set_page_config(
        page_title="BM25 Based Document Ranking - Virtual Lab",
        page_icon="📚",
        layout="wide"
    )
    load_custom_css()

    init_session_state()

    st.title(EXPERIMENT_CONFIG["title"])

st.sidebar.markdown(
    """
    <div style="
        font-size: 1.35rem;
        font-weight: 750;
        color: white;
        margin-bottom: 0.2rem;
    ">
        📚 BM25 Virtual Lab
    </div>

    <div style="
        font-size: 0.8rem;
        color: #94a3b8;
        margin-bottom: 1rem;
    ">
        Document Ranking Experiment
    </div>
    """,
    unsafe_allow_html=True
)

section = st.sidebar.radio(
    "Lab Navigator",
    options=[
        "Purpose",
        "Theory",
        "Simulation",
        "Quiz",
        "Report Generation",
        "Certificate",
        "References"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()

st.sidebar.subheader("Progress Tracker")

quiz_status = (
        "Done"
        if st.session_state.get("quiz_submitted", False)
        else "Pending"
    )

st.sidebar.write(f"- **Quiz Status:** {quiz_status}")

if st.session_state.get("quiz_submitted", False):
        quiz_score = st.session_state.get("quiz_score", 0)

        st.sidebar.write(
    f"- **Quiz Score:** "
    f"`{st.session_state.get('quiz_score', 0)} / {QUIZ_DISPLAY_COUNT}`"
)

trials_recorded = len(
        st.session_state.get("trials", [])
    )

st.sidebar.write(
        f"- **Trials Recorded:** {trials_recorded}"
    )

cert_status = (
        "Eligible"
        if is_lab_completed()
        else "Not yet eligible"
    )

st.sidebar.write(
        f"- **Certificate:** {cert_status}"
    )

if section == "Purpose":
        render_purpose_section()
elif section == "Theory":
        render_theory_section()
elif section == "Simulation":
        render_simulation_section()
elif section == "Quiz":
        render_quiz_section()
elif section == "Report Generation":
        render_report_section()
elif section == "Certificate":
        render_certificate_section()
elif section == "References":
        render_references_section()


if __name__ == "__main__":
    main()
