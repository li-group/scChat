import pytest
from scchatbot.workflow.nodes.planning import PlannerNode

# ---- Shared defaults for tests ----
DEFAULT_AVAILABLE_CELL_TYPES = [
    "Immune cell",
    "B cell",
    "T cell",
    "Natural killer cell",
    "Conventional dendritic cell",
    "Monocyte",
    "Erythroid progenitor cell",
    "Hematopoietic stem cell",
    "Progenitor cell",
    "Stem cell",
]

# --- Helper functions ---

def parse_required_functions(items):
    """
    Accept both dict items like:
      {"function_name": "...", "cell_type": "..."}
    and string items like:
      "process_cells (Immune cell)"
    """
    out = []
    for d in items or []:
        if isinstance(d, dict):
            fn = d.get("function_name")
            ct = d.get("cell_type")
            if fn and ct:
                out.append((fn, ct))
        elif isinstance(d, str):
            if "(" in d and d.endswith(")"):
                fn = d[: d.rfind("(")].strip()
                ct = d[d.rfind("(") + 1 : -1].strip()
                if fn and ct:
                    out.append((fn, ct))
    return out


def build_state(question_text: str, qid: str | None = None):
    """
    Minimal ChatState stub that satisfies PlannerNode expectations.
    """
    return {
        "qid": qid,
        "context": {},
        "current_message": question_text,  # string (node_base uses [:100])
        "messages": [{"role": "user", "content": question_text}],
        "execution_plan": {"steps": []},
        "available_cell_types": DEFAULT_AVAILABLE_CELL_TYPES,
        "function_history_summary": "",     # needed by planner_node
        "selected_cell_types": [],
        "selected_functions": [],
    }


# --- Fixtures ---

@pytest.fixture(scope="module")
def questions():
    import json
    with open("tests/all_questions_hematopoietic.json") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def planner():
    return PlannerNode()


# --- Actual tests ---

QIDS = ["Q7", "Q8", "Q9", "Q10", "Q11", "Q14", "Q15", "Q16", "Q17", "Q18"]

@pytest.mark.parametrize("qid", QIDS)
def test_planner_matches_required_functions_subset(questions, planner, qid):
    q = next((q for q in questions if q.get("question_id") == qid), None)
    if q is None:
        pytest.skip(f"{qid} not present in questions JSON")

    expected = parse_required_functions(q.get("required_functions", []))
    state = build_state(q["question_text"], qid=qid)
    out_state = planner.execute(state)

    # Extract actual (fn, ct) pairs from returned plan
    actual_pairs = []
    for step in out_state.get("execution_plan", {}).get("steps", []):
        fn = step.get("function")
        ct = step.get("args", {}).get("cell_type")
        if fn and ct:
            actual_pairs.append((fn, ct))

    for pair in expected:
        assert pair in actual_pairs, f"Missing required pair {pair} in {qid}"


def test_planner_all_questions_report(questions, planner):
    total, good = 0, 0
    for q in questions:
        expected_pairs = parse_required_functions(q.get("required_functions", []))
        if not expected_pairs:
            continue

        total += 1
        state = build_state(q["question_text"], qid=q.get("question_id"))
        out_state = planner.execute(state)

        actual_pairs = []
        for step in out_state.get("execution_plan", {}).get("steps", []):
            fn = step.get("function")
            ct = step.get("args", {}).get("cell_type")
            if fn and ct:
                actual_pairs.append((fn, ct))

        if all(pair in actual_pairs for pair in expected_pairs):
            good += 1

    print(f"\nPlanner coverage: {good}/{total} questions passed")
    assert good >= int(0.7 * total), f"Too few matches: {good}/{total}"