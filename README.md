# LangGraph Learning

Hands-on examples building agentic workflows with LangGraph — from basic state machines to conditional routing and human-in-the-loop patterns.

## Why LangGraph?

Modern AI applications need more than single LLM calls. They need:

- **Multi-step reasoning**: Chain multiple operations together
- **Conditional branching**: Route based on LLM outputs or business logic
- **Human approval gates**: Pause workflows for review, resume after approval
- **State persistence**: Save progress, handle long-running processes

LangGraph provides the primitives to build these workflows cleanly.

## What's Covered

| File | Concepts |
|------|----------|
| `basic_graph.py` | State, nodes, edges — the fundamentals |
| `conditional_graph.py` | Routing with `add_conditional_edges` |
| `ai_workflow.py` | Full LLM integration with Pydantic extraction |
| `human_in_loop.py` | Interrupt, wait for approval, resume |

## Key Concepts Demonstrated

### 1. State + Nodes + Edges
```python
from langgraph.graph import StateGraph, START, END

class WorkflowState(TypedDict):
    document: str
    classification: str

def classify(state: WorkflowState) -> WorkflowState:
    return {"classification": "loan_disclosure"}

workflow = StateGraph(WorkflowState)
workflow.add_node("classify", classify)
workflow.add_edge(START, "classify")
workflow.add_edge("classify", END)

app = workflow.compile()
result = app.invoke({"document": "...", "classification": ""})
```

### 2. Conditional Routing
```python
def route_by_type(state) -> Literal["extract_loan", "extract_appraisal", "unknown"]:
    if state["classification"] == "loan_disclosure":
        return "extract_loan"
    elif state["classification"] == "appraisal":
        return "extract_appraisal"
    return "unknown"

workflow.add_conditional_edges(
    "classify",
    route_by_type,
    {
        "extract_loan": "extract_loan",
        "extract_appraisal": "extract_appraisal",
        "unknown": "handle_unknown"
    }
)
```

### 3. LLM + Pydantic Integration
```python
class LoanData(BaseModel):
    borrower_name: str
    loan_amount: float

def extract_loan_data(state: DocumentState) -> DocumentState:
    response = completion(model="gpt-4o-mini", messages=[...])
    data = LoanData.model_validate_json(response.choices[0].message.content)
    return {"extracted_data": data.model_dump()}
```

### 4. Human-in-the-Loop
```python
memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["finalize"])

# Start workflow — pauses before "finalize"
result = app.invoke(initial_state, config)

# Human reviews and approves
app.update_state(config, {"approved": True})

# Resume workflow
final = app.invoke(None, config)
```

## Quick Start
```bash
# Clone the repo
git clone https://github.com/georgenavarro1802/langgraph-learning.git
cd langgraph-learning

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env with your API keys

# Run examples
python basic_graph.py
python conditional_graph.py
python ai_workflow.py
python human_in_loop.py
```

## Environment Variables

Create a `.env` file with your API keys (needed for `ai_workflow.py`):
```
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
```

## Requirements

- Python 3.11+
- API keys for OpenAI/Anthropic (for AI workflow examples)

## Next Steps

This repo pairs well with:
- [fastapi-learning](https://github.com/georgenavarro1802/fastapi-learning) - Modern API development
- [litellm-learning](https://github.com/georgenavarro1802/litellm-learning) - Multi-provider LLM abstraction

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)

## License

MIT - Feel free to use, modify, and learn from this code.
