

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pathlib import Path
import re
import os
import dotenv
from storage import Storage
import json

dotenv.load_dotenv()


provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
model = GoogleModel('gemini-2.5-flash', provider=provider)
agent = Agent(
    model,
    system_prompt="""
        You are an expert policy document parser for NebulaGears. Your only job is to read the provided company document and split it into logical, self-contained sections/chunks with rich metadata attached — perfectly suited for a conflict-aware RAG system.

        ### Core Splitting Rules
        - Use headings, sub-headings, FAQs, bullet sections, policy blocks, and natural paragraph groups as boundaries.
        - Never break a single policy rule or sentence across chunks.
        - Target chunk size: 150–450 words (~300 ideal). Merge tiny sections; split very long ones intelligently.
        - Preserve all original formatting clues (e.g., quotes, bold policy statements).
        - Combine all the attached files and generate a single json exactly in the same format below.
        ### Exact Output Format (STRICT — return ONLY this JSON structure)
        {
        "sections": [
            {
            "content": "Full raw text of this section (clean, no markdown artifacts)",
            "metadata": {
                "document_version": "2024.0",
                "effective_date": "2024-06-01",
                "author_department": "People & Culture & Early Careers Program",
                "document_type": "intern_faq",                 // "handbook" | "policy_update" | "intern_faq"
                "applies_to": "interns, contractors",          // string: possible values: "all_employees", "full_time_employees", "interns", "contractors", "managers". (If multiple, comma-separated)
                "policy_topic": "office_presence",             // e.g. remote_work, onboarding, mentorship, pto, security, etc.
                "specificity_level": "role_specific",          // "general" | "role_specific" | "department_specific"
                "is_role_specific_intern": true,
                "is_authoritative_for_interns": true,          // ← true for ALL chunks from intern_onboarding_faq.txt
                "supersedes_older_policies": false,
                "conflict_resolution_note": true,              // true if section mentions conflict resolution, authoritative source, etc.
            }
            },
            { ... next section ... }
        ]
        }

        ### Mandatory Tagging Logic
        - Every chunk from **intern_onboarding_faq.txt** → `is_authoritative_for_interns = true`
        - Any chunk that explicitly mentions "intern(s)", "internship", "entry-level participant" → 
        - `applies_to` includes "interns"
        - `is_role_specific_intern = true`
        - `specificity_level = "role_specific"`
        - Chunks containing phrases like "in case of conflict", "authoritative source", "refer to this document", "this document intentionally specifies" → `conflict_resolution_note = true`
        - manager_updates_2024.txt chunks that update/restrict remote work → `supersedes_older_policies = true`
        - Use the exact heading (if exists) as `chunk_title`, otherwise generate a short clear one.

        ### Final Instruction
        Return ONLY the valid JSON object above. No explanations, no markdown, no extra text whatsoever.
    """,
)


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

async def main():
    paths = [Path("Assignment/employee_handbook_v1.txt"), Path("Assignment/manager_updates_2024.txt"), Path("Assignment/intern_onboarding_faq.txt")]
    response = await agent.run(
        [
            """
            Split the following company policy documents into meaningful chunks with rich metadata as per the provided instructions.
            """,
            
            *[BinaryContent(data=file_.read_bytes(), media_type="text/plain") for file_ in paths]
        ]
    )
    print(response.output)

    response_json = force_raw_json(response.output)

    # save json to file for inspection
    with open("data_chunks.json", "w") as f:
        json.dump(response_json, f, indent=4)

    storage = Storage(collection_name="nebula_gears_policies")
    storage.add_document(
            document_contents=[section["content"] for section in response_json["sections"]],
            metadatas=[section["metadata"] for section in response_json["sections"]],
        )



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())