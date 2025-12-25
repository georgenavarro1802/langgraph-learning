from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class ApprovalState(TypedDict):
    document: str
    extracted_amount: float
    requires_approval: bool
    approved: bool
    status: str


def extract_amount(state: ApprovalState) -> ApprovalState:
    """Simulate extraction of the amount"""
    amount = 500000.00
    requires_approval = amount > 100000.0   # Huge amounts require approval

    print(f"   [extract_amount] -> amount=${amount:,.0f}, requires approval={requires_approval}")
    return {
        "extracted_amount": amount,
        "requires_approval": requires_approval
    }

def check_approval_needed(state: ApprovalState) -> Literal["wait_approval", "auto_approve"]:
    """Check if needs human approval"""
    if state["requires_approval"]:
        return "wait_approval"
    return "auto_approve"

def wait_for_approval(state: ApprovalState) -> ApprovalState:
    """Node that whait for human approval"""
    print(f"   [wait_approval] -> Waiting for human approval...")
    # The current State is saved, the workflow is paused here
    return {"status": "waiting"}

def auto_approve(state: ApprovalState) -> ApprovalState:
    """Automatically approve small amounts"""
    print(f"   [auto_approve] + Auto-approved (amount under threshold)")
    return {"approved": True, "status": "auto_approved"}

def finalize(state: ApprovalState) -> ApprovalState:
    """Finalize the process"""
    if state["approved"]:
        print(f"   [finalize] + Processing Complete!")
        return {"status": "complete"}
    else:
        print(f"   [finalize] - Rejected")
        return {"status": "rejected"}

# Build the graph
workflow = StateGraph(ApprovalState)

# Add Nodes
workflow.add_node("extract", extract_amount)
workflow.add_node("wait_approval", wait_for_approval)
workflow.add_node("auto_approve", auto_approve)
workflow.add_node("finalize", finalize)

# Add Edges (conections)
workflow.add_edge(START, "extract")
workflow.add_conditional_edges(
    "extract",
    check_approval_needed,
    {
        "wait_approval": "wait_approval",
        "auto_approve": "auto_approve"
    }
)
workflow.add_edge("wait_approval", "finalize")
workflow.add_edge("auto_approve", "finalize")
workflow.add_edge("finalize", END)

# Checkpointer - store the current state in memory for summary
memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["finalize"])

# Simulate execution with interrupts
print("=" * 50)
print("HUMAN-IN-THE-LOOP EXECUTION")
print("=" * 50)

# Thread ID to track this conversation
config = {"configurable": {"thread_id": "loan-123"}}

# Step 1 - Start Workflow
print("\n--- Step 1: Start Workflow ---")
initial_state = {
    "document": "Loan for $500,000",
    "extracted_amount": 0.0,
    "requires_approval": False,
    "approved": False,
    "status": ""
}
result = app.invoke(initial_state, config=config)
print(f"State after extraction: {result}")
print(f"Status: {result['status']}")

# The workflow is paused here before the "finalize"
# In a real app, we would show the user a UI here for approval.

# Step 2: Simulate human approval
print("\n--- Step 2: Human approves ---")
# we update here the status with the human decision
app.update_state(config, {"approved": True})

# Step 3: Summarize the workflow
print("\n--- Step 3: Resume workflow ---")
final_result = app.invoke(None, config=config)
print(f"Final state: {final_result}")