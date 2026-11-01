from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import planner_node, router_node, conversation_node, image_generation_node, image_edit_node, image_analysis_node


def build_graph():
    graph = StateGraph(AgentState)
    for name, node in [
        ("planner", planner_node), ("router", router_node),
        ("conversation", conversation_node), ("image_generation", image_generation_node),
        ("image_edit", image_edit_node), ("image_analysis", image_analysis_node),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "router")
    graph.add_conditional_edges("router", lambda s: s.get("route", "conversation"), {
        "conversation": "conversation", "image_generation": "image_generation",
        "image_edit": "image_edit", "image_analysis": "image_analysis",
    })
    for name in ("conversation", "image_generation", "image_edit", "image_analysis"):
        graph.add_edge(name, END)
    return graph.compile()


graph = build_graph()
