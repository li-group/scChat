# tests/test_planner_node.py
import json
import pytest

from scchatbot.workflow.nodes.planning import PlannerNode


class _FakeResp:
    def __init__(self, content: str):
        self.content = content


class _SmartFakeChatOpenAI:

    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, messages):
        # messages is a list of SystemMessage/HumanMessage; we inspect the last human prompt
        prompt = ""
        for m in messages[::-1]:
            if hasattr(m, "content") and isinstance(m.content, str):
                prompt = m.content
                break

        if "Create a plan in this JSON format" in prompt:
            plan = {
                "plan_summary": "Direct response",
                "visualization_only": False,
                "steps": [
                    {
                        "step_type": "conversation",
                        "function_name": "direct_answer",
                        "parameters": {},
                        "description": "say hi",
                        "expected_outcome": "text"
                    }
                ],
            }
            return _FakeResp(json.dumps(plan))

        if "Extract biological pathway-related keywords" in prompt:
            # Simulate extracting a meaningful keyword
            return _FakeResp("interferon response")

        # Fallback for unexpected prompts
        return _FakeResp("")


def _mk_planner():
    """Build a PlannerNode without running BaseWorkflowNode.__init__."""
    p = object.__new__(PlannerNode)
    # Minimal attributes used by the planner methods we test
    p.initial_cell_types = []
    p.function_descriptions = []
    p.hierarchy_manager = None
    p.enrichment_checker_available = False
    p.enrichment_checker = None
    return p


def test_planner_returns_execution_plan(monkeypatch: pytest.MonkeyPatch):
    """
    End-to-end of _create_enhanced_plan with an LLM stub that returns a simple 'direct_answer' plan.
    Ensures the planner writes an 'execution_plan' into the state.
    """
    # Stub the model used inside PlannerNode
    monkeypatch.setattr(
        "scchatbot.workflow.nodes.planning.ChatOpenAI",
        _SmartFakeChatOpenAI,
        raising=True,
    )

    planner = _mk_planner()
    state = {
        "current_message": "Hi",
        "available_cell_types": [],
        "function_history_summary": {},
        "messages": [],
        "has_conversation_context": False,
    }

    out = planner._create_enhanced_plan(
        state,
        state["current_message"],
        planner.function_descriptions,
        state["available_cell_types"],
        [],
    )

    assert "execution_plan" in out
    steps = out["execution_plan"]["steps"]
    assert len(steps) == 1
    assert steps[0]["function_name"] == "direct_answer"
    assert out["execution_plan"]["plan_summary"] == "Direct response"


def test_skip_unavailable_cell_steps():
    """
    Pure unit test for filtering steps by unavailable cell types.
    """
    planner = _mk_planner()
    plan = {
        "steps": [
            {"description": "work on T", "parameters": {"cell_type": "T cell"}},
            {"description": "work on B", "parameters": {"cell_type": "B cell"}},
        ]
    }

    filtered = planner._skip_unavailable_cell_steps(plan, ["B cell"])
    steps = filtered["steps"]
    assert len(steps) == 1
    assert steps[0]["parameters"]["cell_type"] == "T cell"


def test_enrichment_step_is_enhanced_with_keywords(monkeypatch: pytest.MonkeyPatch):
    """
    Verify that an enrichment step is enhanced with analyses/gene_set_library
    when pathway keywords are extracted and enrichment_checker is unavailable.
    """
    # Avoid importing/using cell_types.validation by short-circuiting discovery
    monkeypatch.setattr(
        PlannerNode,
        "_add_cell_discovery_to_plan",
        lambda self, plan_data, message, available_cell_types: plan_data,
        raising=True,
    )

    # Stub the ChatOpenAI used for pathway keyword extraction
    monkeypatch.setattr(
        "scchatbot.workflow.nodes.planning.ChatOpenAI",
        _SmartFakeChatOpenAI,
        raising=True,
    )

    planner = _mk_planner()

    # Minimal enrichment plan produced by the (stubbed) planner
    plan_data = {
        "plan_summary": "Run enrichment",
        "visualization_only": False,
        "steps": [
            {
                "step_type": "analysis",
                "function_name": "perform_enrichment_analyses",
                "parameters": {"cell_type": "T cell"},
                "description": "Run enrichment for T cell",
                "expected_outcome": "enrichment results"
            }
        ]
    }

    enhanced = planner._process_plan(
        plan_data,
        message="Please analyze interferon pathways in T cells",
        available_cell_types=["T cell"],
        unavailable_cell_types=[],
    )

    # The first step should have been enhanced with defaults since enrichment_checker is unavailable
    s0 = enhanced["steps"][0]
    assert s0["function_name"] == "perform_enrichment_analyses"
    params = s0.get("parameters", {})
    assert params.get("analyses") == ["gsea"]  # set by fallback path when keywords are present
    assert params.get("gene_set_library") == "MSigDB_Hallmark_2020"