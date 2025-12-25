from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from litellm import completion
from dotenv import load_dotenv
import json

load_dotenv()


# Pydantic models for structured extraction
class LoanData(BaseModel):
    borrower_name: str = Field(..., description="Name of the borrower")
    loan_amount: float = Field(..., description="Loan amount in dollars")
    interest_rate: float = Field(..., description="Interest rate as percentage")


class AppraisalData(BaseModel):
    property_address: str = Field(..., description="Property address")
    appraised_value: float = Field(..., description="Appraised value in dollars")


# Workflow state
class DocumentState(TypedDict):
    document: str
    classification: str
    extracted_data: dict
    confidence: float
    error: str


def classify_with_llm(state: DocumentState) -> DocumentState:
    """Use LLM to classify the document"""
    prompt = f"""Classify this document into one of these categories:
- loan_disclosure
- appraisal
- unknown

Document: 
{state['document']}

Reply with ONLY the category name, nothing else.
"""

    response = completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    classification = response["choices"][0]["message"]["content"].strip().lower()
    print(f"   [classify_llm] -> {classification}")

    return {"classification": classification}

def extract_loan_data(state: DocumentState) -> DocumentState:
    """Extract loan data from loan disclosure documents"""
    schema = LoanData.model_json_schema()

    prompt = f"""Extract loan information from this document as JSON.
    
Document: 
{state['document']}

Schema:
{json.dumps(schema, indent=2)}

Return ONLY valid JSON.
"""

    response = completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    json_str = response["choices"][0].message.content
    json_str = json_str.replace("```json", "").replace("```", "").strip()

    data = LoanData.model_validate_json(json_str)
    print(f"   [extract_loan_data] -> {data.model_dump()}")

    return {"extracted_data": data.model_dump(), "confidence": 0.95}

def extract_appraisal_data(state: DocumentState) -> DocumentState:
    """Extract appraisal data from appraisal documents"""
    schema = AppraisalData.model_json_schema()

    prompt = f"""Extract appraisal information from this document as JSON.

Document: 
{state['document']}

Schema:
{json.dumps(schema, indent=2)}

Return ONLY valid JSON.
"""

    response = completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    json_str = response["choices"][0].message.content
    json_str = json_str.replace("```json", "").replace("```", "").strip()

    data = AppraisalData.model_validate_json(json_str)
    print(f"   [extract_appraisal_data] -> {data.model_dump()}")

    return {"extracted_data": data.model_dump(), "confidence": 0.92}

def handle_unknown(state: DocumentState) -> DocumentState:
    """Handle unknown document type"""
    print(f"   [handle_unknown] -> Needs human review")
    return {"error": "Unrecognized document type. Please review manually.", "confidence": 0.0}

def route_by_type(state: DocumentState) -> Literal["classify_llm", "extract_loan_data", "extract_appraisal_data", "handle_unknown"]:
    """Route to the correct function based on the document type"""
    if state["classification"] == "loan_disclosure":
        return "extract_loan"
    elif state["classification"] == "appraisal":
        return "extract_appraisal"
    return "unknown"

# Build the graph
workflow = StateGraph(DocumentState)

# Add Nodes
workflow.add_node("classify", classify_with_llm)
workflow.add_node("extract_loan", extract_loan_data)
workflow.add_node("extract_appraisal", extract_appraisal_data)
workflow.add_node("unknown", handle_unknown)

# Add Edges (conections)
workflow.add_edge(START, "classify")
workflow.add_conditional_edges(
    "classify",
    route_by_type,
    {
        "extract_loan": "extract_loan",
        "extract_appraisal": "extract_appraisal",
        "unknown": "unknown"
    }
)
workflow.add_edge("extract_loan", END)
workflow.add_edge("extract_appraisal", END)
workflow.add_edge("unknown", END)

# Compile Workflow
app = workflow.compile()

# Tests
print("=" * 50)
print("TEST 1: Loan Document")
print("=" * 50)
loan_doc = """
LOAN DISCLOSURE STATEMENT
Borrower: Maria Pons
Loan Amount: $325,000
Annual Interest Rate: 5.875%
Term: 30 years
Lender: First National Bank
"""
result1 = app.invoke({
    "document": loan_doc,
    "classification": "",
    "extracted_data": {},
    "confidence": 0.0,
    "error": ""
})
print(f"\nFinal: {result1}\n")

print("=" * 50)
print("TEST 2: Appraisal Document")
print("=" * 50)
appraisal_doc = """
PROPERTY APPRAISAL REPORT
Property Address: 123 Main St, New York, NY 10001
Appraised Value: $500,000
Appraisal Date: Decemeber 20, 2025
Appraiser: George Navarro
"""
result2 = app.invoke({
    "document":appraisal_doc,
    "classification": "",
    "extracted_data": {},
    "confidence": 0.0,
    "error": ""
})
print(f"\nFinal: {result2}\n")

print("=" * 50)
print("TEST 3: Unknown Document")
print("=" * 50)
unknown_doc = """
RESTAURANT MENU
Pizza: $12.99
Burger: $10.99
Drinks: $5.99
"""
result3 = app.invoke({
    "document": unknown_doc,
    "classification": "",
    "extracted_data": {},
    "confidence": 0.0,
    "error": ""
})
print(f"\nFinal: {result3}\n")