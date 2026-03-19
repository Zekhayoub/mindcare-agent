# MindCare — Carbon-Aware Emotional Support Agent

An AI agent that routes emotional support queries between a local ML classifier and a cloud LLM, minimizing carbon footprint while maximizing response quality. Built for SDG 3 (Good Health and Well-Being) as part of the AI course at ECAM Brussels.

> **Architecture:** ML Pre-Analysis → Confidence-Based Routing → ECO (Local) or AGENT (Cloud) → CO2 Tracking

> **Stack:** Python, scikit-learn, LangChain, Mistral AI, FAISS, Streamlit, Folium, tiktoken, NLTK

---

## Why This Project Matters

* **The Access Problem:** Mental health services are saturated, expensive, and stigmatized. MindCare provides **24/7 emotional triage**, not a replacement for therapists, but a first line of informational support while waiting for professional help.

* **The Carbon Problem:** Every LLM query costs energy. Instead of sending everything to the cloud, MindCare uses a **hybrid routing strategy** that processes ~80% of queries locally with a lightweight ML classifier, reserving the cloud LLM for complex or ambiguous cases. This cuts CO2 emissions by ~80%.

* **The Safety Problem:** Generic LLMs can hallucinate medical advice. MindCare uses **deterministic safety guards** (keyword detection before any generation) and **RAG grounding** (responses anchored in verified clinical documents) to prevent dangerous outputs.

---

## Demo

### ECO Mode — Local Processing

<p align="center"><img src="figures/demo_eco_mode.png" width="800"></p>

* **Emotion detected:** Sadness (97% confidence) → routed to **ECO mode** (no LLM call)
* **Personalized advice** from the enriched CSV database (context: social, intensity: severe)
* **Clinical excerpt** retrieved via RAG from verified psychology sources
* **CO2 cost:** 0.70 g — displayed in the technical expander with full token breakdown

### AGENT Mode — Crisis Response

<p align="center"><img src="figures/demo_crisis_response.png" width="800"></p>

* **Safety trigger:** danger words detected → immediately routed to **AGENT mode** (Mistral Cloud)
* **Crisis protocol:** breathing technique (4-4-6), emergency numbers (Samaritans UK: 116 123, 999)
* **Location-aware:** detects "London" from user input, suggests nearby safe location with Folium map
* **Geolocation fallback:** if location detection fails, the UI asks the user directly

### Activity Recommendation

<p align="center"><img src="figures/demo_activity_map.png" width="300"></p>

* **End-of-session feature:** suggests a real location based on dominant emotion and detected city
* **Nominatim API** for geocoding with config-based fallback locations
* **Interactive Folium map** embedded in the Streamlit sidebar

---

## Architecture

```
User Input
    │
    ▼
┌──────────────────────┐
│  ML Pre-Analysis     │  Local, <5ms
│  (LogReg + TF-IDF)   │  Emotion + Confidence
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  The Strategist      │  Rules: safety, complexity,
│  (Confidence Router) │  confidence threshold (60%)
└────┬────────────┬────┘
     │            │
  ECO Mode     AGENT Mode
  (Local)      (Mistral Cloud)
     │            │
     ▼            ▼
  CSV Advice   LangChain ReAct
  + RAG        + Tool Calling
  + Activity   + Full Reasoning
     │            │
     └─────┬──────┘
           ▼
  Response + CO2 Tracking
```

The Strategist checks five rules in order:
1. **Safety keywords** → always escalate to AGENT
2. **Question complexity** → questions need LLM reasoning
3. **Classifier confidence** → below 60% means uncertain, escalate
4. **Unknown emotion** → classifier could not decide, escalate
5. **Emotion type** → some emotions need nuanced handling

If all checks pass → **ECO mode** (local, fast, green). Otherwise → **AGENT mode** (cloud, powerful, higher CO2).

---

## Results

### Emotion Classifier (Test Set — 62,519 samples)

| Emotion  | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Sadness  | 0.98      | 0.93   | 0.95     | 18,178  |
| Joy      | 0.98      | 0.89   | 0.93     | 21,159  |
| Love     | 0.74      | 0.96   | 0.84     | 5,183   |
| Anger    | 0.89      | 0.94   | 0.92     | 8,597   |
| Fear     | 0.88      | 0.86   | 0.87     | 7,156   |
| Surprise | 0.67      | 0.92   | 0.78     | 2,246   |
| **Overall** | **0.92** | **0.91** | **0.91** | **62,519** |

* **Winner:** Logistic Regression (F1=0.914) over SVM (0.909) and Naive Bayes (0.862), chosen for the best **accuracy-to-cost ratio**
* **class_weight='balanced'** to handle class imbalance (Joy/Sadness dominate, Surprise underrepresented)
* **TF-IDF vectorizer** fit only on the training set to prevent data leakage
* **Weakness:** Fear/Surprise confusion — both share vocabulary like "trembling", which a bag-of-words approach cannot disambiguate

<p align="center"><img src="figures/confusion_matrix_champion.png" width="600"></p>
<p align="center"><em>Confusion matrix — strong diagonal, expected Fear/Surprise overlap</em></p>

### Qualitative Benchmark (LLM-as-a-Judge)

| Category               | Baseline (Mistral) | MindCare Agent |
|------------------------|--------------------|----------------|
| Safety & Crisis        | 4/5                | **5/5**        |
| Clinical Precision     | 3/5                | **4/5**        |
| Local Knowledge        | 3/5                | **4/5**        |
| Nuance & Negation      | 4/5                | 4/5            |

* **Safety score 5/5** — deterministic keyword guards trigger before LLM generation
* **Clinical precision** — RAG provides grounded techniques (Behavioral Activation, 5-4-3-2-1 grounding) instead of hallucinated advice
* **Local knowledge** — geolocation returns real, verifiable places, not invented addresses

---

## CO2 Impact

| Architecture      | CO2 per query  | Annual (5K users/day) |
|-------------------|----------------|-----------------------|
| Full Cloud (LLM)  | ~0.17 g        | ~316 kg               |
| MindCare Hybrid   | ~0.03 g        | ~63 kg                |
| **Savings**       |                | **~253 kg (80%)**     |

* **253 kg CO2 saved annually** — equivalent to driving 2,100 km in a car
* **Dominant optimization factor:** algorithmic routing, not datacenter geography — "green coding" matters more than "green infrastructure"
* **Token output length** has the highest weight on footprint, validating our prompt engineering strategy: concise responses = eco-design
* **GHG Protocol:** Scope 1 = 0 (software only), Scope 2 = local CPU (negligible), Scope 3 = Mistral API (reduced 80% by routing)

---

## Exploratory Data Analysis

<p align="center"><img src="figures/emotion_distribution.png" width="600"></p>
<p align="center"><em>Class distribution — Joy and Sadness dominate, Surprise underrepresented</em></p>

* **416,793 rows** from the Kaggle Emotions dataset, 6 classes
* **Class imbalance** motivated the use of `class_weight='balanced'` in training
* **Short messages** (median ~15 words) — confirms that TF-IDF + LogReg is appropriate, heavy transformers would be overkill

<p align="center"><img src="figures/word_count_by_emotion.png" width="600"></p>
<p align="center"><em>Message length by emotion — short texts across all classes</em></p>


---




## Approach

1. **Data cleaning** — Semantic-aware NLP pipeline preserving negations ("not happy" stays meaningful). Lemmatization, custom stopword removal. 416K rows cleaned with 16 dropped.

2. **Training** — Stratified 70/15/15 split. TF-IDF (n-grams 1-3, 20K features, fit on train only). Tournament: NB vs LogReg vs SVM. Winner: LogReg (F1=0.914, `class_weight='balanced'`).

3. **RAG** — Clinical PDFs chunked (800 chars, 100 overlap), embedded with Mistral, indexed in FAISS (142 chunks). Provides grounded clinical techniques to prevent hallucination.

4. **Agent** — LangChain ReAct with 6 tools: emotion analysis, advice retrieval, activity suggestion, clinical manual, GPS lookup, knowledge search. Few-shot prompt with crisis and standard examples.

5. **CO2 tracking** — Token-level estimation using Mistral LCA factors (0.00285 gCO2e/token). Uncertainty band ±20%. Per-query display in the technical expander.

---

## Data

* **Emotions dataset:** 416,793 short English text messages from Kaggle, 6 classes (Joy, Sadness, Love, Anger, Fear, Surprise)
* **Advice database:** 68 curated entries with emotion/intensity/context/technique/citation, sourced from clinical psychology guides
* **RAG sources:** 4 PDFs + 1 TXT — coping skills, grounding techniques, relaxation skills, stress workbook

---

## References

- United Nations, "THE 17 GOALS | Sustainable Development", sdgs.un.org
- Xi et al., "The rise and potential of large language model based agents: A survey", 2023
- LangChain Documentation, python.langchain.com
- Meta Research, "FAISS: Efficient similarity search", Meta AI
- GHG Protocol, "Calculation Tools for Direct and Indirect Emissions", ghgprotocol.org
- Nelgiriyewithana, "Emotions", Kaggle
- Mistral AI, "Our contribution to a global environmental standard for AI", mistral.ai