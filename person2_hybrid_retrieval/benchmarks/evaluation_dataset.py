"""
Controlled Evaluation Dataset for ProofMesh Hybrid Retrieval (Person 2 Workstream).
Contains a diverse 15-chunk reference corpus and 12 curated benchmark queries spanning:
1. Exact keyword queries
2. Paraphrase / conceptual queries
3. Numerical queries
4. Date queries
5. Entity / structured ID queries
6. Policy / legal wording queries
7. Technical / hardware queries
8. Out-of-context / unanswerable queries
"""

from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Evaluation Corpus (15 Document Chunks across Legal, HR, Hardware, SLA domains)
# ---------------------------------------------------------------------------
EVALUATION_CORPUS: List[Dict[str, Any]] = [
    {
        "chunk_id": "chunk_01",
        "doc_name": "payment_policy.pdf",
        "text": "The purchaser shall remit the outstanding invoice within thirty calendar days of receipt.",
        "category": "legal_payment",
    },
    {
        "chunk_id": "chunk_02",
        "doc_name": "billing_spec.pdf",
        "text": "The invoice reference number must be included in all electronic wire payment correspondence.",
        "category": "legal_payment",
    },
    {
        "chunk_id": "chunk_03",
        "doc_name": "warranty_terms.pdf",
        "text": "The hardware warranty remains valid for twenty-four months from the original purchase date.",
        "category": "hardware_warranty",
    },
    {
        "chunk_id": "chunk_04",
        "doc_name": "warranty_terms.pdf",
        "text": "Damage resulting from liquid spills or unauthorized disassembly is expressly excluded from coverage.",
        "category": "hardware_warranty",
    },
    {
        "chunk_id": "chunk_05",
        "doc_name": "hr_handbook.pdf",
        "text": "Employees must submit vacation requests through the HR portal at least two weeks in advance.",
        "category": "hr_operations",
    },
    {
        "chunk_id": "chunk_06",
        "doc_name": "hr_handbook.pdf",
        "text": "Parental leave policy POL-4092 entitles eligible staff to twelve weeks of fully paid leave.",
        "category": "hr_operations",
    },
    {
        "chunk_id": "chunk_07",
        "doc_name": "technical_spec.pdf",
        "text": "Project Hackingly utilizes the Intel Core i5-12450H CPU executing local offline inference under 500 milliseconds.",
        "category": "technical_hardware",
    },
    {
        "chunk_id": "chunk_08",
        "doc_name": "technical_spec.pdf",
        "text": "The device consumes fifteen watts of power under nominal multi-threaded vector workload.",
        "category": "technical_hardware",
    },
    {
        "chunk_id": "chunk_09",
        "doc_name": "sla_agreement.pdf",
        "text": "System uptime SLA is guaranteed at ninety-nine point nine percent excluding scheduled maintenance windows.",
        "category": "sla_terms",
    },
    {
        "chunk_id": "chunk_10",
        "doc_name": "sla_agreement.pdf",
        "text": "Routine server maintenance occurs every Sunday between 02:00 UTC and 04:00 UTC.",
        "category": "sla_terms",
    },
    {
        "chunk_id": "chunk_11",
        "doc_name": "security_audit.pdf",
        "text": "The enterprise security certification ISO-27001 was officially renewed on October 14 2025.",
        "category": "compliance",
    },
    {
        "chunk_id": "chunk_12",
        "doc_name": "security_audit.pdf",
        "text": "All user passwords must be hashed using bcrypt with work factor twelve.",
        "category": "compliance",
    },
    {
        "chunk_id": "chunk_13",
        "doc_name": "returns_policy.pdf",
        "text": "The return window for unopened factory accessories is thirty calendar days with receipt.",
        "category": "returns",
    },
    {
        "chunk_id": "chunk_14",
        "doc_name": "contract_amendment.pdf",
        "text": "Arbitration proceedings shall be conducted exclusively in New Delhi under Indian law.",
        "category": "legal_jurisdiction",
    },
    {
        "chunk_id": "chunk_15",
        "doc_name": "contact_info.pdf",
        "text": "Official support inquiries can be directed to support@proofmesh.ai or telephone 1800-555-0199.",
        "category": "contact",
    },
]

# ---------------------------------------------------------------------------
# Evaluation Queries (12 Items spanning the 8 required categories)
# ---------------------------------------------------------------------------
EVALUATION_QUERIES: List[Dict[str, Any]] = [
    # 1. Exact Keyword Query
    {
        "id": "q01",
        "query": "invoice reference number payment correspondence",
        "expected_chunk_ids": ["chunk_02"],
        "category": "exact_keyword",
        "description": "High lexical overlap with chunk 02.",
    },
    # 2. Paraphrase / Conceptual Query
    {
        "id": "q02",
        "query": "When does the customer need to pay the bill?",
        "expected_chunk_ids": ["chunk_01"],
        "category": "paraphrase",
        "description": "Conceptual equivalence: 'purchaser shall remit invoice' vs 'customer need to pay bill'.",
    },
    # 3. Numerical Query
    {
        "id": "q03",
        "query": "What is the 24 months warranty duration?",
        "expected_chunk_ids": ["chunk_03"],
        "category": "numerical",
        "description": "Numeric constraint '24' and 'warranty'.",
    },
    # 4. Date Query
    {
        "id": "q04",
        "query": "When was ISO-27001 renewed on October 14 2025?",
        "expected_chunk_ids": ["chunk_11"],
        "category": "date",
        "description": "Exact date query matching certification record.",
    },
    # 5. Entity / ID Query
    {
        "id": "q05",
        "query": "What are the rules under policy POL-4092?",
        "expected_chunk_ids": ["chunk_06"],
        "category": "entity_id",
        "description": "Exact structured identifier 'POL-4092'.",
    },
    # 6. Policy / Legal Wording Query
    {
        "id": "q06",
        "query": "Which legal jurisdiction handles arbitration disputes?",
        "expected_chunk_ids": ["chunk_14"],
        "category": "policy_legal",
        "description": "Legal wording matching arbitration clause.",
    },
    # 7. Medical / Technical Terminology Query
    {
        "id": "q07",
        "query": "What is the password hashing algorithm and bcrypt work factor?",
        "expected_chunk_ids": ["chunk_12"],
        "category": "medical_technical",
        "description": "Technical cryptographic terms 'bcrypt' and 'work factor'.",
    },
    # 8. Technical Hardware Query
    {
        "id": "q08",
        "query": "What CPU processor does Project Hackingly use for 500 ms latency?",
        "expected_chunk_ids": ["chunk_07"],
        "category": "medical_technical",
        "description": "Hardware processor and latency target specification.",
    },
    # 9. Paraphrase Leave Query
    {
        "id": "q09",
        "query": "Where do I submit my vacation and timeoff request?",
        "expected_chunk_ids": ["chunk_05"],
        "category": "paraphrase",
        "description": "Conceptual matching for employee HR leave portal.",
    },
    # 10. Numerical Power Query
    {
        "id": "q10",
        "query": "How many watts of power does the device consume?",
        "expected_chunk_ids": ["chunk_08"],
        "category": "numerical",
        "description": "Numerical inquiry matching 'fifteen watts'.",
    },
    # 11. Exact Keyword SLA Query
    {
        "id": "q11",
        "query": "system uptime SLA guaranteed maintenance windows",
        "expected_chunk_ids": ["chunk_09"],
        "category": "exact_keyword",
        "description": "Lexical overlap with service level agreement terms.",
    },
    # 12. Out-of-Context / Unanswerable Query
    {
        "id": "q12",
        "query": "What is the patient blood pressure and cardiovascular medication?",
        "expected_chunk_ids": [],
        "category": "out_of_context",
        "description": "Domain alien query that has zero true positive chunks in the corpus.",
    },
]
