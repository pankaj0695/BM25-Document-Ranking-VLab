# KGIRS Virtual Lab

**Experiment Number:** 5
**Experiment Title:** BM25 Based Document Ranking
**Division:** D17A
**Roll Numbers:** 21–25

**Deployed App:** [https://bm25-document-ranking-vlab-pankaj.streamlit.app/](https://bm25-document-ranking-vlab-pankaj.streamlit.app/)

## About the Project

This virtual lab is an interactive Streamlit application that implements **BM25**, a probabilistic document ranking function, from first principles and compares it against the classic **TF-IDF** vector-space model with cosine similarity. Given a collection of documents and a search query, both methods score every document for relevance and rank them — the lab lets you run both rankers side by side on the same corpus and query and see exactly where and why their rankings diverge.

The core idea it demonstrates: plain TF-IDF has no saturation on term frequency, so a document that simply repeats a query term many times can rank artificially high. BM25 fixes this with two tunable parameters — **k1**, which saturates the contribution of repeated term occurrences, and **b**, which normalizes for document length — so that concise, genuinely relevant documents aren't outranked by long, keyword-stuffed ones. The app makes this concrete with a built-in document corpus (plus the option to paste in a custom one), live TF-IDF vs. BM25 ranking tables, a score comparison chart, and a per-term BM25 contribution breakdown, all updating as you adjust k1 and b.

Beyond the simulation, the lab follows a complete virtual-lab structure:

- **Theory** – the Aim, TF-IDF and BM25 formulas, a comparison table, the experimental procedure, and key terminology
- **Simulation** – the interactive ranking sandbox with a trial log book you can export as CSV
- **Quiz** – a 10-question self-graded assessment on TF-IDF and BM25 concepts, with instant feedback
- **Report Generation** – compiles your recorded trials and quiz score into a downloadable PDF lab report

Built with Python, Streamlit, Plotly, and fpdf2, with no external dependencies for the ranking logic itself — TF-IDF and BM25 are both implemented from scratch using standard-library tokenization, term/document frequency counting, and the BM25 formula.
