# Intelligent Complaint Analysis for Financial Services

## RAG-Powered Customer Complaint Analysis

An end-to-end Retrieval-Augmented Generation (RAG) project for analyzing financial-services customer complaints using semantic search, sentence embeddings, and a local language model.

The project is designed around a practical business problem: **helping product, support, compliance, and risk teams identify recurring customer problems without manually reading thousands of complaint narratives.**

---

## 📌 Project Overview

Financial institutions receive large volumes of customer complaints through digital channels. These complaints contain valuable information about customer frustrations, product weaknesses, recurring service failures, and emerging issues—but extracting those insights manually is slow and difficult.

This project explores how a **Retrieval-Augmented Generation (RAG)** system can transform unstructured complaint narratives into an interactive question-answering system.

A user can ask questions such as:

> **"Why are customers unhappy with credit cards?"**

The system:

1. Converts the question into an embedding.
2. Searches complaint embeddings using cosine similarity.
3. Retrieves the most relevant complaint chunks.
4. Builds a context from the retrieved evidence.
5. Sends the context and question to a language model.
6. Generates an answer grounded in the retrieved complaint narratives.
7. Exposes the retrieved sources so that the answer can be inspected.

---

## 🎯 Business Objective

The project is based on a hypothetical financial institution, **CrediTrust Financial**, serving customers through a mobile-first platform.

The business objective is to help internal teams:

* Identify major complaint trends faster.
* Understand recurring customer pain points.
* Search complaint narratives using natural language.
* Reduce dependence on manual complaint analysis.
* Support evidence-based product decisions.
* Give non-technical stakeholders easier access to customer feedback.

### Target users

* Product Managers
* Customer Support teams
* Compliance teams
* Risk teams
* Business analysts
* Financial-services decision makers

---

## 🏗️ System Architecture

The implemented workflow follows the RAG architecture:

```text
                    ┌──────────────────────┐
                    │   Customer Complaints │
                    │      CFPB Dataset     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Data Cleaning &     │
                    │        EDA            │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Text Chunking      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Sentence Embeddings  │
                    │ all-MiniLM-L6-v2     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Vector Embeddings  │
                    └──────────┬───────────┘
                               │
                    User Question
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Query Embedding      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Cosine Similarity    │
                    │ Retrieval            │
                    └──────────┬───────────┘
                               │
                         Top-k Chunks
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Prompt Construction  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ FLAN-T5 Generator    │
                    │ Local CPU Inference  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Answer + Sources     │
                    └──────────────────────┘
```

---

# 📊 Dataset

The project uses complaint data from the **Consumer Financial Protection Bureau (CFPB)**.

The dataset contains information such as:

* Complaint identifiers
* Product categories
* Issue categories
* Sub-issues
* Company information
* Consumer complaint narratives
* Dates
* Geographic metadata

The project focuses on financial products relevant to the challenge, including:

* Credit Cards
* Personal Loans
* Savings Accounts
* Money Transfers

---

# 🔎 Task 1 — Exploratory Data Analysis & Preprocessing

The first stage focused on understanding and preparing the complaint dataset for downstream NLP tasks.

### Work completed

The EDA and preprocessing workflow includes:

* Loading the complaint dataset.
* Inspecting the dataset structure.
* Examining complaint categories.
* Investigating consumer complaint narratives.
* Checking missing narrative values.
* Examining narrative length.
* Filtering relevant complaint records.
* Removing complaints without usable narratives.
* Cleaning complaint text.
* Preparing the data for chunking and embedding.

### Notebook

```text
notebooks/
└── 01_eda.ipynb
```

The notebook documents the exploratory analysis and preprocessing process.

---

# 🧩 Task 2 — Text Chunking & Embeddings

Long complaint narratives are not always suitable for representing as a single embedding.

Therefore, the project uses a chunking pipeline to break complaint narratives into smaller pieces before generating embeddings.

### Embedding model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model was selected because it provides a practical balance between:

* Semantic representation quality
* Computational efficiency
* Embedding size
* Local CPU usability

The model generates **384-dimensional embeddings**.

### Embedding workflow

```text
Complaint Narrative
        ↓
Text Chunking
        ↓
Individual Complaint Chunks
        ↓
all-MiniLM-L6-v2
        ↓
384-dimensional vectors
        ↓
Embedding dataset
```

### Notebook

```text
notebooks/
└── 02_chunking_embeddings.ipynb
```

The notebook contains the chunking and embedding workflow.

---

# 🔍 Task 3 — RAG Pipeline

The project implements the core components of a Retrieval-Augmented Generation system.

### Retrieval

When a user submits a question, the system:

1. Encodes the question with `all-MiniLM-L6-v2`.
2. Compares the query embedding against complaint embeddings.
3. Calculates cosine similarity.
4. Selects the top-k most similar complaint chunks.
5. Returns the complaint text, metadata, and similarity score.

The current retrieval implementation uses:

```python
k = 5
```

as the default number of retrieved chunks.

### Retrieval result

Each retrieved result contains:

```text
text
metadata
similarity score
```

This allows the generated answer to remain connected to the complaint evidence.

---

# 🧠 Prompt Engineering

The generation stage uses a prompt designed to reduce unsupported answers.

The model is instructed to:

* Act as a financial analyst assistant.
* Use only the retrieved complaint excerpts.
* Identify common themes.
* Identify customer frustrations.
* Identify recurring issues.
* Avoid inventing information.
* State when the retrieved context does not contain enough information.

This is important because financial complaint analysis requires **evidence-backed responses rather than unsupported speculation**.

---

# 🤖 Language Model

For local generation, the project uses:

```text
google/flan-t5-base
```

The model runs locally through Hugging Face Transformers.

The implementation was intentionally tested using CPU inference to accommodate environments without dedicated GPU resources.

### Generation workflow

```text
Retrieved Complaints
        +
User Question
        ↓
Prompt Template
        ↓
FLAN-T5
        ↓
Generated Answer
```

---

# 🧪 RAG Evaluation

A set of representative questions was used to test the retrieval and generation workflow.

The evaluation considers:

* Relevance of retrieved complaints
* Quality of generated answers
* Evidence grounding
* Whether the response addresses the question
* Potential hallucinations
* Overall usefulness

Evaluation results are stored in:

```text
evaluation.csv
```

The RAG development and evaluation work is documented in:

```text
notebooks/
└── 03_rag_pipeline.ipynb
```

---

# 🖥️ Interactive Application

A Streamlit application was developed as the user-facing interface.

The intended user experience is:

```text
┌────────────────────────────────────────────┐
│     CrediTrust Complaint Assistant         │
├────────────────────────────────────────────┤
│                                            │
│ Ask a question about customer complaints:  │
│                                            │
│ [ Why are customers unhappy with cards? ]  │
│                                            │
│                  [ Ask ]                   │
│                                            │
├────────────────────────────────────────────┤
│ Answer                                     │
│                                            │
│ Generated response based on retrieved      │
│ complaint evidence.                        │
│                                            │
├────────────────────────────────────────────┤
│ Retrieved Sources                           │
│                                            │
│ Source 1                                   │
│ Source 2                                   │
│ Source 3                                   │
└────────────────────────────────────────────┘
```

The application entry point is:

```text
app.py
```

Run the application with:

```bash
streamlit run app.py
```

---

# 📁 Project Structure

```text
rag-complaint-chatbot/
│
├── .github/
│   └── workflows/
│
├── .vscode/
│   └── settings.json
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_chunking_embeddings.ipynb
│   └── 03_rag_pipeline.ipynb
│
├── src/
│   ├── __init__.py
│   ├── retriever.py
│   └── rag.py
│
├── tests/
│
├── app.py
├── evaluation.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🛠️ Technology Stack

| Area                 | Technology                |
| -------------------- | ------------------------- |
| Programming Language | Python                    |
| Data Processing      | Pandas, NumPy             |
| Data Analysis        | Jupyter Notebook          |
| NLP                  | Sentence Transformers     |
| Embedding Model      | all-MiniLM-L6-v2          |
| Similarity Search    | Cosine Similarity         |
| Generation           | FLAN-T5                   |
| LLM Framework        | Hugging Face Transformers |
| Interface            | Streamlit                 |
| Version Control      | Git & GitHub              |

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Helina-Leul/rag-complaint-chatbot.git
cd rag-complaint-chatbot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

### Run the notebooks

Open the project in VS Code and select the project's virtual environment/kernel.

Run the notebooks in order:

```text
01_eda.ipynb
       ↓
02_chunking_embeddings.ipynb
       ↓
03_rag_pipeline.ipynb
```

### Run the Streamlit application

From the project root:

```bash
streamlit run app.py
```

---

# ⚠️ Important Data & Hardware Note

The full complaint embedding dataset is very large.

The pre-built embedding file used during development is approximately **2.24 GB**, and loading the complete dataset into memory can exceed the available RAM on a standard personal computer.

For this reason, the large embedding artifact is intentionally **not stored in this GitHub repository**.

This is a deliberate engineering decision rather than an omission.

The repository contains the code and notebooks needed to understand and reproduce the processing and RAG workflow, while large generated artifacts are excluded through `.gitignore`.

A production implementation should use a persistent vector database such as FAISS or ChromaDB rather than loading the complete embedding matrix into RAM.

---

# 🧱 Current Project Status

This repository represents the work completed during development under the available computing and time constraints.

### Completed

* [x] Project structure
* [x] CFPB complaint data exploration
* [x] Data preprocessing
* [x] Complaint filtering
* [x] Text cleaning
* [x] Text chunking
* [x] Sentence-transformer embeddings
* [x] 384-dimensional complaint embeddings
* [x] Semantic retrieval
* [x] Cosine similarity search
* [x] Top-k complaint retrieval
* [x] Metadata retrieval
* [x] Context construction
* [x] RAG prompt template
* [x] Local FLAN-T5 generation
* [x] RAG pipeline implementation
* [x] Initial qualitative evaluation
* [x] Evaluation dataset
* [x] Streamlit application structure
* [x] Git/GitHub version control

### Not fully completed / Production improvements

The following areas remain opportunities for future development:

* [ ] Production-scale vector database deployment
* [ ] Memory-efficient retrieval over the complete dataset
* [ ] More extensive RAG evaluation
* [ ] Retrieval metrics such as Recall@K and MRR
* [ ] Automated evaluation
* [ ] More robust hallucination testing
* [ ] Production deployment
* [ ] Authentication and access control
* [ ] Response streaming
* [ ] Advanced multi-product comparison
* [ ] Automated monitoring and logging

Being explicit about these limitations is important because the goal of this repository is to accurately represent the engineering work completed rather than claim production readiness where it has not yet been achieved.

---

# 💡 Key Engineering Lessons

This project provided practical experience with several important AI engineering concepts.

### 1. Data quality comes before AI

A RAG system is only as useful as the information it retrieves. Cleaning and understanding complaint narratives is therefore an essential first step.

### 2. Retrieval quality directly affects generation quality

Even a strong language model cannot reliably answer a question if the retriever provides irrelevant evidence.

### 3. Embeddings enable semantic search

Traditional keyword search may miss complaints that use different wording to describe the same problem.

Embedding-based retrieval allows semantically similar complaints to be identified even when the exact words differ.

### 4. Evidence matters

Displaying retrieved sources makes the system more transparent and allows users to inspect the evidence behind an answer.

### 5. Hardware constraints affect architecture

Running large-scale NLP models and millions of embeddings locally can create significant memory and performance constraints.

This project demonstrated why production RAG systems need careful consideration of:

* Vector databases
* Indexing strategies
* Memory usage
* Model size
* Retrieval latency
* Deployment infrastructure

---

# 🚀 Future Improvements

A production-ready version could improve the current implementation through:

### Vector database

Replace the in-memory cosine similarity approach with:

```text
FAISS
```

or:

```text
ChromaDB
```

This would make retrieval more scalable.

### Better retrieval

Introduce:

* Metadata filtering
* Product filtering
* Hybrid keyword + semantic retrieval
* Re-ranking
* Retrieval confidence thresholds

### Better generation

Evaluate smaller and more capable instruction-tuned models and compare:

* Answer quality
* Latency
* Memory consumption
* Hallucination rate

### Better evaluation

Create a larger benchmark containing:

```text
Question
Expected evidence
Retrieved evidence
Generated answer
Grounding score
Relevance score
Completeness score
Overall score
```

### Production deployment

A future version could be deployed as:

```text
Streamlit / React UI
        ↓
FastAPI backend
        ↓
RAG service
        ↓
Vector Database
        ↓
LLM
```

with authentication, logging, monitoring, and automated testing.

---

# 📌 Conclusion

This project demonstrates the complete foundation of a financial complaint RAG system—from **raw customer feedback to semantic retrieval and evidence-grounded generation**.

The most important outcome is not simply generating an answer with an LLM. The project demonstrates the complete reasoning pipeline:

```text
Customer Complaints
        ↓
Data Understanding
        ↓
Preprocessing
        ↓
Chunking
        ↓
Embeddings
        ↓
Semantic Retrieval
        ↓
Evidence
        ↓
Prompt Construction
        ↓
LLM Generation
        ↓
Answer + Sources
```

The project also highlights an important engineering reality: **a working prototype and a production-ready system are not the same thing.**

The current implementation establishes the core RAG architecture while clearly documenting the scalability, memory, evaluation, and deployment improvements required for a production environment.

---

## 👩‍💻 Author

**Helina Leul**

Software Engineering Student | Data & AI Engineering Learner

Interests:

* Data Analytics
* Machine Learning
* Natural Language Processing
* Retrieval-Augmented Generation
* AI Engineering
* Financial Technology

---

## ⭐ Project Focus

**Turning unstructured financial customer complaints into searchable, evidence-backed insights using RAG.**
