from storage import Storage
from pydantic_ai import Agent, ImageUrl, BinaryContent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
import re
import json
import os

filter_llm_prompt = """
    You are NebulaGears Query Filter Extractor. Your only job is to analyze the user's question and return ONLY a valid ChromaDB `where` filter that guarantees the most authoritative, role-specific documents are retrieved first — especially ensuring intern-specific rules win when the user is an intern.

    Use EXACTLY these metadata fields and values (do not invent new ones):

    - document_type: "handbook" | "policy_update" | "intern_faq"
    - applies_to: containing any of: "all_employees", "full_time_employees", "interns", "contractors", "managers". (If multiple, comma-separated ex: "interns, contractors")
    - policy_topic: string (e.g. "remote_work", "office_presence", "onboarding", etc.)
    - specificity_level: "general" | "role_specific" | "department_specific"
    - is_role_specific_intern: true | false
    - is_authoritative_for_interns: true | false
    - supersedes_older_policies: true | false
    - conflict_resolution_note: true | false

    ### Detection Rules (strict)
    - If user mentions: intern, internship, "new intern", "just joined as intern", "summer intern", "co-op", "entry-level" → user is intern
    - Remote work keywords ("work from home", "remote", "wfh", "office", "in-office", "hybrid") → topic is remote_work or office_presence

    ### Priority Logic
    1. If user is an intern → MUST include is_authoritative_for_interns = true (this alone beats everything)
    2. Always boost role-specific over general
    3. For interns: intern_faq chunks must win even if cosine similarity favors older docs

    ### Output Format — ONLY this JSON, nothing else
    Return exactly one of these structures:

    # For intern asking about remote work
    {
    "$or": [
        { "is_authoritative_for_interns": true },
        { "is_role_specific_intern": true },
        { "applies_to": "interns" }
    ]
    }

    # For full-time employee asking about remote work
    {
    "$or": [
        { "document_type": "policy_update" },
        { "supersedes_older_policies": true },
        { "applies_to": { "$in": ["all_employees", "full_time_employees"] } }
    ]
    }

    # Broad fallback (unknown role)
    {
    "policy_topic": { "$in": ["remote_work", "office_presence"] }
    }

    Return ONLY the raw JSON filter object. No variable name, no markdown, no explanation, no extra text.

"""

def force_raw_json(response: str) -> dict:
    response = response.strip()
    # Remove ```json ... ``` or ``` ... ```
    response = re.sub(r"^```json\s*\n", "", response, flags=re.IGNORECASE)
    response = re.sub(r"\n```$", "", response)
    response = re.sub(r"^```\s*\n", "", response)
    response = re.sub(r"\n```$", "", response)
    # If it still has whitespace or garbage, find the first { and last }
    start = response.find("{")
    end = response.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON found")
    json_str = response[start:end]
    return json.loads(json_str)


async def smart_retrieve(collection_name: str, question: str, n_results: int = 20):

    """
        Smart document retrieval with dynamic filtering based on user question
    """

    filter_prompt = filter_llm_prompt + f'\n\nuser_question: {question}\n\nProvide only the JSON filter:'

    provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
    model = GoogleModel('gemini-2.5-flash', provider=provider)
    agent = Agent(
        model=model,
        system_prompt="You are an expert at generating database query filters based on user questions.",
    )

    result = await agent.run(filter_prompt)
    # print(result.output)
    filter_json = force_raw_json(result.output)
    # print(filter_json)
    try:
        where_filter = filter_json
        if where_filter == {}:
            where_filter = None
    except:
        where_filter = None

    # Step 3: Query with smart filtering
    storage = Storage(collection_name=collection_name)
    collection = storage.chroma_client.get_collection(collection_name, storage.embedding_function)
    results = collection.query(
        query_texts=[question],
        n_results=n_results * 3,  # get more, then rerank
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    # Step 4: Simple reranking by distance
    combined = list(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]))
    combined.sort(key=lambda x: x[2])  # sort by distance
    top_docs = [doc for doc, meta, dist in combined[:n_results]]

    return top_docs, where_filter