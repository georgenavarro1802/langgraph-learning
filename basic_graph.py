from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. Define the State - What data travels through the graph
class WorkflowState(TypedDict):
    document: str
    classification: str
    extracted_data: dict
    is_valid: bool


# 2. Define Nodes - functions that transform the state
def classify_document(state: WorkflowState) -> WorkflowState:
    """Classify the document type"""
    doc = state['document'].lower()

    if "loan" in doc:
        classification = "loan_disclosure"
    elif "appraisal" in doc:
        classification = "appraisal"
    else:
        classification = "unknown"

    print(f"   [classify] -> {classification}")
    # return {**state, 'classification': classification}
    return {"classification": classification}

def extract_data(state: WorkflowState) -> WorkflowState:
    """Extract data from the document"""
    # simulation - here make a call to LLM for real
    extracted = {
        "type": state['classification'],
        "has_content": len(state['document']) > 0
    }

    print(f"   [extract] -> {extracted}")
    return {"extracted_data": extracted}

def validate_data(state: WorkflowState) -> WorkflowState:
    """Validate the extracted data"""
    is_valid = (
        state["classification"] != "unknown"
        and state["extracted_data"].get("has_content", False)
    )

    print(f"   [validate] -> valid={is_valid}")
    return {"is_valid": is_valid}

# 3. Build the graph
workflow = StateGraph(WorkflowState)

# Add Nodes
workflow.add_node("classify", classify_document)
workflow.add_node("extract", extract_data)
workflow.add_node("validate", validate_data)

# Add Edges (conections)
workflow.add_edge(START, "classify")
workflow.add_edge("classify", "extract")
workflow.add_edge("extract", "validate")
workflow.add_edge("validate", END)

# Compile
app = workflow.compile()

# 4. Execute
print("=== Running workflow ===\n")
initial_state = {
    "document": "This is a LOAN disclosure document for $500,000",
    "classification": "",
    "extracted_data": {},
    "is_valid": False
}

result = app.invoke(initial_state)

print("\n=== Final State ====")
print(result)
