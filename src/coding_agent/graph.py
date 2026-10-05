from langgraph.graph import END, START, StateGraph

from coding_agent.nodes import AgentNodes
from coding_agent.state import AgentState


def build_graph(llm, checkpointer=None):
    nodes = AgentNodes(llm)
    g = StateGraph(AgentState)

    g.add_node("scan_repo", nodes.scan_repo)
    g.add_node("select_files", nodes.select_files)
    g.add_node("read_files", nodes.read_files)
    g.add_node("make_plan", nodes.make_plan)
    g.add_node("plan_approval", nodes.plan_approval)
    g.add_node("generate_changes", nodes.generate_changes)
    g.add_node("build_diff", nodes.build_diff_node)
    g.add_node("explain", nodes.explain)

    g.add_edge(START, "scan_repo")
    g.add_edge("scan_repo", "select_files")
    g.add_edge("select_files", "read_files")
    g.add_edge("read_files", "make_plan")
    g.add_edge("make_plan", "plan_approval")
    g.add_conditional_edges(
        "plan_approval",
        nodes.route_after_plan,
        {"generate": "generate_changes", "stop": END},
    )
    g.add_edge("generate_changes", "build_diff")
    g.add_edge("build_diff", "explain")
    g.add_edge("explain", END)

    return g.compile(checkpointer=checkpointer)