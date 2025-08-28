# tests/test_planner_e2e_real.py
import os
import sys
import json
import pytest
from pathlib import Path

# ---- Config / constants -------------------------------------------------------
QIDS = ["Q7","Q8","Q9","Q10","Q11","Q14","Q15","Q16","Q17","Q18"]
QUESTIONS_JSON = os.environ.get(
    "PLANNER_QUESTIONS_JSON",
    str(Path(__file__).resolve().parent / "data" / "all_questions_hematopoietic.json"),
)
REQUIRE_API = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; set it to run real E2E planner tests.",
)

# ---- Small helpers copied from the other test for scoring ---------------------
from typing import Dict, List, Tuple, Any
def parse_required_functions(items: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    out = []
    for d in items or []:
        fn = d.get("function_name")
        cell = d.get("cell_type")
        if fn and cell:
            out.append((fn, cell))
    return out

def extract_plan_calls(state: Dict[str, Any]) -> List[Tuple[str, str]]:
    plan = (state.get("execution_plan") or {}).get("steps", [])
    pairs = []
    for step in plan:
        fn = (step.get("call") or {}).get("function")
        args = (step.get("call") or {}).get("args") or {}
        cell = args.get("cell_type") or args.get("target_cell_type") or args.get("group")
        if fn and cell:
            pairs.append((fn, cell))
    return pairs

def score_plan(expected: List[Tuple[str, str]], got: List[Tuple[str, str]]):
    exp = set(expected)
    got_s = set(got)
    tp = len(exp & got_s)
    fp = len(got_s - exp)
    fn = len(exp - got_s)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return type("MatchResult", (), {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "missing": list(exp - got_s), "extras": list(got_s - exp)
    })()

def build_state(user_text: str, qid: str | None = None) -> Dict[str, Any]:
    return {
        "user_query": user_text,
        "qid": qid,
        "execution_plan": {"steps": []},
        "context": {},
    }

# ---- Fixtures ----------------------------------------------------------------
@pytest.fixture(scope="session")
def questions():
    with open(QUESTIONS_JSON, "r") as f:
        data = json.load(f)
    # normalize to list of question dicts
    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]
    return data

@pytest.fixture()
@REQUIRE_API
def planner(monkeypatch):
    """
    Real PlannerNode (real LLM), but monkeypatch BaseWorkflowNode.__init__
    to avoid heavy construction. We *do not* mock the LLM.
    """
    # Ensure repo imports resolve
    repo_root = Path(__file__).resolve().parents[1]
    for p in (repo_root, repo_root / "src"):
        sp = str(p)
        if p.exists() and sp not in sys.path:
            sys.path.insert(0, sp)

    # Import after sys.path fixups
    import scchatbot.workflow.node_base as node_base  # correct location
    from scchatbot.workflow.nodes.planning import PlannerNode

    # Lightweight shim for BaseWorkflowNode.__init__
    def _lite_init(self, *args, **kwargs):
        # seed the attributes PlannerNode expects to exist
        self.initial_annotation_content = ""
        self.initial_cell_types = []
        self.adata = None
        self.history_manager = kwargs.get("history_manager") or object()
        self.hierarchy_manager = kwargs.get("hierarchy_manager") or object()
        self.cell_type_extractor = kwargs.get("cell_type_extractor") or (lambda *_: [])
        # Minimal function metadata used by the planner
        self.function_descriptions = {
            "dea_split_by_condition": "DEA per cell type across conditions.",
            "compare_cell_counts": "Compare counts between groups for a cell type.",
            "perform_enrichment_analyses": "Run enrichment analyses for a cell type.",
            "display_enrichment_visualization": "Visualize enrichment for a cell type.",
            "process_cells": "Preprocess / subset cells by type."
        }
        self.function_mapping = {k: k for k in self.function_descriptions}
        self.visualization_functions = {"display_enrichment_visualization": "display_enrichment_visualization"}
        # Provide a small curated list the planner can pick from
        self.valid_cell_types = [
            "T cell", "B cell", "Immune cell", "Progenitor cell", "Erythroid progenitor cell"
        ]
        # Any other lightweight flags the node might check
        self.neo4j_manager = None
        self.embedding_model = None

    monkeypatch.setattr(node_base.BaseWorkflowNode, "__init__", _lite_init, raising=True)

    # Environment toggles for the node to use real API
    os.environ["PLANNER_E2E"] = "1"
    os.environ.setdefault("PLANNER_MODEL", os.environ.get("PLANNER_MODEL", "gpt-4o-mini"))

    return PlannerNode()

# ---- Tests -------------------------------------------------------------------
@pytest.mark.parametrize("qid", QIDS)
@REQUIRE_API
def test_planner_matches_required_functions_subset(questions, planner, qid):
    # Skip if the particular qid doesn't exist in the chosen questions file
    q = next((q for q in questions if q.get("question_id") == qid), None)
    if q is None:
        pytest.skip(f"{qid} not found in provided questions JSON; skipping.")

    expected = parse_required_functions(q.get("required_functions", []))
    state = build_state(q["question_text"], qid=qid)
    out_state = planner.execute(state)
    got = extract_plan_calls(out_state)

    s = score_plan(expected, got)
    assert s.precision >= 0.5 or s.recall >= 0.5, (
        f"Low match for {qid}: P={s.precision:.2f} R={s.recall:.2f} "
        f"missing={s.missing} extras={s.extras}"
    )

@REQUIRE_API
def test_planner_all_questions_report(questions, planner):
    total = 0
    good = 0
    for q in questions:
        state = build_state(q["question_text"], qid=q.get("question_id"))
        out_state = planner.execute(state)
        steps = (out_state.get("execution_plan") or {}).get("steps", [])
        assert isinstance(steps, list), "execution_plan.steps should be a list"

        expected_pairs = parse_required_functions(q.get("required_functions", []))
        if expected_pairs:
            got = extract_plan_calls(out_state)
            s = score_plan(expected_pairs, got)
            total += 1
            if s.precision >= 0.5 or s.recall >= 0.5:
                good += 1

    if total:
        assert good / total >= 0.5, f"Overall loose match rate too low: {good}/{total}"