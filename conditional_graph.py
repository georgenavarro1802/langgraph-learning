from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


class WorkflowState(TypedDict):
    document: str
    classification: str
    extracted_data: dict
    error_message: str


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
    return {"classification": classification}

def extract_data(state: WorkflowState) -> WorkflowState:
    """Extract data from the document"""
    print(f"   [extract] -> Processing {state['classification']}")
    return {
        "extracted_data": {
            "type": state['classification'],
            "processed": True
        }
    }

def handle_unknown(state: WorkflowState) -> WorkflowState:
    """Handle unknown classification"""
    print(f"   [handle_unknown] -> Needs human review")
    return {
        "error_message": "Document type not recognized. Please review mannually."
    }

# Decision Function - determine wich path to take
def route_by_classification(state: WorkflowState) -> Literal["extract", "handle_unknown"]:
    if state["classification"] == "unknown":
        return "handle_unknown"
    return "extract"


# Build the graph
workflow = StateGraph(WorkflowState)

# Add Nodes
workflow.add_node("classify", classify_document)
workflow.add_node("extract", extract_data)
workflow.add_node("handle_unknown", handle_unknown)

# Add Edges (conections)
workflow.add_edge(START, "classify")

# Conditional edge - here the magic happens
workflow.add_conditional_edges(
    "classify",                 # from this node
    route_by_classification,    # use this funtion to decide which path to take
    {
        "extract": "extract",   # if returns "extract" go to "extract" node"
        "handle_unknown": "handle_unknown" # if returns "handle_unknown" go to "handle_unknown" node
    }
)


workflow.add_edge("extract", END)
workflow.add_edge("handle_unknown", END)

# Compile
app = workflow.compile()

# Test with valid document
print("=== Test 1: Valid document ===")
result1 = app.invoke({
    "document": "LOAN disclosure for $500,000",
    "classification": "",
    "extracted_data": {},
    "error_message": ""
})
print(f"Result: {result1}\n")

# Test with unknown document
print("\n=== Test 2: Unknown document ===")
result2 = app.invoke({
    "document": "Random document about something else",
    "classification": "",
    "extrated_data": {},
    "error_message": ""
})
print(f"Result: {result2}\n")