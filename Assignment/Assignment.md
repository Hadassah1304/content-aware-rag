# GenAI Intern Assessment: The "Conflict-Aware" RAG System

**Stack:** Google Gemini + ChromaDB (Local)

### **The Context**

In the real world, companies don't just want AI; they want **cost-effective** AI that runs on their own data.You are tasked with building a local RAG (Retrieval Augmented Generation) system that answers employee questions. However, the company documents contain contradictory information.

**Your Mission:** Build a RAG pipeline that detects policy conflicts and prioritizes the correct information based on the user's role.

---

### **The Tech Stack (Strict Requirements)**

You must use the following free/open-source tools:

1.  **LLM:** **Google Gemini Flash 2.5**.
    -   *Why:* It has a generous free tier via Google AI Studio.
2.  **Vector Database:** **ChromaDB** (or FAISS).
    -   *Why:* It must run locally (no cloud vector stores like Pinecone).
3.  **Embeddings:** **Google GenAI Embeddings** (`models/text-embedding-004`) or `SentenceTransformers` (HuggingFace).

---

### **The Dataset**

You are provided with 3 text chunks representing the knowledge base of a fictional company, *NebulaGears*.

**Document A: `employee_handbook_v1.txt`**

> "At NebulaGears, we believe in complete freedom. All employees are eligible for the 'Work From Anywhere' program. You can work remotely 100% of the time from any location. No prior approval is needed."

**Document B: `manager_updates_2024.txt`**

> "Update to remote work policy: Effective immediately, remote work is capped at 3 days per week. Employees must be in the HQ office on Tuesdays and Thursdays. All remote days require manager approval."

**Document C: `intern_onboarding_faq.txt`**

> "Welcome to the team! Please note that while full-time employees have hybrid options, **interns are required to be in the office 5 days a week** for the duration of their internship to maximize mentorship. No remote work is permitted for interns."

---

### **The Task**

Create a Python script that ingests these documents into ChromaDB and answers questions using Gemini.

**The Scenario:**A user queries: *"I just joined as a new intern. Can I work from home?"*

**The "Impossible" Constraints:**A standard RAG system using Cosine Similarity will likely retrieve Document A or B because they share many keywords with the query ("work", "home", "remote"), leading the AI to give the wrong answer.

**To pass, your system must:**

1.  **Retrieve Correctly:** Ensure Document C (Intern FAQ) is prioritized or successfully retrieved alongside the others.
2.  **Reason Correctly:** The final answer must state that while general employees have hybrid options, **interns** strictly cannot work from home.
3.  **Cite Sources:** The output must explicitly mention which document provided the final ruling.

---

### **The Deliverables**

1.  **The Code:** A GitHub repository with a clean Python script.
    -   Must include `requirements.txt` (e.g., `langchain`, `chromadb`, `google-generativeai`).
    -   Must utilize the Gemini API Key (instructions on how to set it up).
2.  **The "Conflict Logic":**
    -   In your code or README, explain how you solved the conflict.
    -   *Did you use Metadata Filtering?* (e.g., Tagging docs as "General" vs "Intern").
    -   *Did you use a "Re-ranking" step?*
    -   *Did you use a specific prompt to force Gemini to analyze dates/specificity?*
3.  **Cost Analysis:**
    -   Include a brief note: If we scaled this to 10,000 documents and 5,000 queries a day, roughly how much would this architecture cost using Gemini Flash?

### Bonus Points ✅

-   **Open-source LLMs:** If you use any open-source model (e.g., Llama 2, Mistral, Falcon, RedPajama, etc.) for your RAG pipeline instead of (or along with) Gemini Flash, you'll receive a bonus point in the submission review.
-   **No Local GPU? Use Colab:** If you don’t have sufficient local compute to run these open-source models, you can run them on **Google Colab** or similar free notebook platforms — include the Colab link and usage instructions in your README.
-   **Documentation:** If you use an open-source model, mention it clearly in your README and explain any additional configuration (e.g., quantization, smaller model variants, or HF transformers) you used to make the model run efficiently.

---

### **Instructions for the Candidate**

1.  Get a free API Key from [Google AI Studio](https://aistudio.google.com/).
2.  Install ChromaDB (`pip install chromadb`).
3.  Write the solution.
4.  Submit the GitHub Link + a screenshot of the output for the "Intern" query.

---