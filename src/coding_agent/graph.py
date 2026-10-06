from langgraph.graph import END, START, StateGraph

from coding_agent.nodes import AgentNodes
from coding_agent.state import AgentState


def build_graph(llm, checkpointer=None):
    nodes = AgentNodes(llm)
    g = StateGraph(AgentState)

    g.add_node("input_guard", nodes.input_guard)
    g.add_node("output_guard", nodes.output_guard)
    g.add_node("scan_repo", nodes.scan_repo)
    g.add_node("select_files", nodes.select_files)
    g.add_node("read_files", nodes.read_files)
    g.add_node("make_plan", nodes.make_plan)
    g.add_node("expand_files", nodes.expand_files)
    g.add_node("plan_approval", nodes.plan_approval)
    g.add_node("generate_changes", nodes.generate_changes)
    g.add_node("build_diff", nodes.build_diff_node)
    g.add_node("explain", nodes.explain)
    g.add_node("run_tests", nodes.run_tests_node)
    g.add_node("apply_approval", nodes.apply_approval)
    g.add_node("apply_changes", nodes.apply_changes_node)

    g.add_edge(START, "input_guard")
    g.add_edge("input_guard", "scan_repo")
    g.add_edge("scan_repo", "select_files")
    g.add_edge("select_files", "read_files")
    g.add_edge("read_files", "make_plan")
    g.add_conditional_edges(
        "make_plan",
        nodes.route_after_draft,
        {"expand": "expand_files", "approve": "plan_approval"},
    )
    g.add_edge("expand_files", "read_files")
    g.add_conditional_edges(
        "plan_approval",
        nodes.route_after_plan,
        {"generate": "generate_changes", "stop": END},
    )
    g.add_edge("generate_changes", "output_guard")
    g.add_edge("output_guard", "run_tests")
    g.add_conditional_edges(
        "run_tests",
        nodes.route_after_tests,
        {"retry": "generate_changes", "finish": "build_diff"},
    )
    g.add_edge("build_diff", "explain")
    g.add_edge("explain", "apply_approval")
    g.add_conditional_edges(
        "apply_approval",
        nodes.route_after_apply,
        {"apply": "apply_changes", "stop": END},
    )
    g.add_edge("apply_changes", END)

    return g.compile(checkpointer=checkpointer)