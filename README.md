# TaxStreem Internship — AI / ML Track - Intelligent Transaction Grouper

## Approach

I chose the **Hybrid approach (Option C)** — combining semantic embeddings with LLM-powered label generation.

**Why hybrid over pure LLM or pure embeddings:**
A pure LLM approach sends all transactions directly to the model for grouping and labeling in one shot. This works for small inputs but becomes expensive and less reliable at scale. The model handles both the mathematical grouping and the semantic labeling, making errors harder to isolate.

A pure embedding approach clusters transactions by vector similarity efficiently and cheaply, but cannot generate meaningful labels or explanations. It only sees numbers, not meaning.

The hybrid pipeline separates these concerns deliberately:
- **sentence-transformers** handles mathematical similarity locally (free, deterministic, no API cost)
- **Groq LLaMA 3.3** handles semantic labeling and explanation (small, targeted API call per run)

Each tool does only what it is best at. This is more reliable, more cost-efficient at scale, and easier to debug than either approach alone.

---

## Prompt Design
The labeling prompt is structured in five sections:
1. **Role and context** — establishes the Nigerian fintech context so the model recognises local companies like Paystack, Flutterwave, MTN, and Airtel correctly
2. **Numbered instructions** — four explicit tasks with no ambiguity
3. **Rules** — pre-handles known edge cases: the Uber vs Uber Eats distinction, confidence level definitions, singleton group handling
4. **Strict JSON instruction** — prevents the model from wrapping output in markdown code fences which would break parsing
5. **Few-shot example** — provides a concrete input/output example to anchor the expected format.

---

## Assumptions
- Transaction descriptions are short strings (under 20 words)
- The input is a flat JSON array of strings
- All transactions are in a Nigerian fintech context

---
## Output
Outputs are saved as output.json in the data folder (data/output.json).

## Trade-offs

**What my solution gets wrong:**

1. **KMeans requires a fixed k** — I estimate k as the square root of half the input size. For 16 transactions this produces k=3, which is too low and causes unrelated items to be forced into the same cluster. In the sample output, MTN/Airtel and Shoprite ended up in the same cluster, and Uber Eats merged with Uber rides. This project will still require more tuning.

2. **Paystack and Flutterwave in separate groups** — these are both Nigerian payment processors and should be one group. The embedding model placed them far enough apart that KMeans separated them. Domain-specific fine-tuning of the embedding model would fix this.

3. **LLM labels are non-deterministic** — I tested and confirmed that running the same input twice may produce slightly different labels, explanations and confidence level. The embedding clustering is deterministic (random_state=42) but the LLM labeling is not. 

4. **Deliberate restraint on parameter tuning** — The expected output shows 7 groups with specific labels. During development I experimented with different values of k and prompt variations. Some produced worse results, some came closer to the expected 7 groups. I deliberately stopped short of tuning specifically toward the expected output because doing so would constitute data leakage — I would be optimising for a known answer rather than building a system that generalises to unseen transaction data. The parameters I settled on represent my best judgment for a general-purpose grouping system, not a system fitted to this specific sample.

**What would break it:**
- Transactions in languages other than English
- Very large inputs (1000+ transactions) would hit free API rate limits. 
- Highly ambiguous or irregular descriptions with no recognisable merchant names or names that LLM or transformers can easily find.

---

## Evaluation

**How I would measure grouping quality without ground truth labels:**
1. **Intra-cluster cosine similarity** — measure the average cosine similarity between all items within each cluster. High similarity = tight, meaningful cluster. This is computable without any labels.

2. **Silhouette score** — scikit-learn provides this out of the box. It measures how similar each item is to its own cluster versus other clusters. Score ranges from -1 to 1, higher is better.

3. **Human spot-check rate** — sample 10% of grouped transactions and have a human verify correctness. Track this rate over time as a quality signal.

4. **Singleton rate** — a high percentage of singleton groups suggests k is too high or the embedding model is not capturing semantic similarity well enough.

---

## How to Run

**1. Clone the repository**
git clone https://github.com/YOUR_USERNAME/internship_test.git
cd internship_test/ai_ml_track

**2. Install dependencies**
pip install -r requirements.txt

**3. Set up environment variables**
cp .env.example .env
Edit `.env` and add your Groq API key:
GROQ_API_KEY=your_groq_api_key_here
Get a free Groq API key at https://console.groq.com

**4. Run the pipeline**
python src/main.py

**5. Run the tests**
python -m pytest tests/ -v

Output is printed to the terminal and saved to `data/output.json`.

## Project Structure
```
ai_ml_track/
├── src/
│   ├── embeddings.py        # Loads sentence-transformer model, 
│                            # converts descriptions to vectors
│   ├── grouper.py           # Core pipeline: clustering + LLM labeling
│   ├── prompt_templates.py  # Prompt engineering for LLM labeling
│   └── main.py              # Entry point, input loading, output saving
├── tests/
│   └── test_grouper.py      # Unit tests for parsing and post-processing
├── data/
│   ├── sample_input.json    # Input transactions
│   └── output.json          # Generated output (created on run)
├── requirements.txt
├── .env.example
└── README.md
```
---

## Cost Estimate

**Per 1,000 transactions:**
- Embedding (sentence-transformers): **$0.00** — runs locally on CPU
- Groq LLaMA 3.3 70B: approximately **$0.06** per 1,000 transactions (based on ~600 input tokens per cluster summary × estimated 15 clusters per 1,000 transactions × $0.59/million tokens)

**Total: approximately $0.06 per 1,000 transactions**
This is significantly cheaper than a pure LLM approach which would send all raw transactions directly, costing roughly $0.40-0.80 per 1,000 transactions depending on description length.

---

## Dependencies

| Library | Purpose |
|---|---|
| sentence-transformers | Local CPU embedding model |
| scikit-learn | KMeans clustering, silhouette scoring |
| groq | LLM API client for label generation |
| python-dotenv | Environment variable management |

No LangChain or high-level AI frameworks were used. All LLM interactions are implemented directly against the Groq API to maintain full visibility and control over every step of the pipeline.
