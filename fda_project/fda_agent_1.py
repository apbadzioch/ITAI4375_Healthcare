"""
fda_agent.py  —  LangGraph ReAct agent for the FDA Drug Label Assistant

Architecture:
  - StateGraph with MessagesState (built-in message list + add_messages reducer)
  - Two nodes: 'agent' (ChatOllama with bound tools) and 'tools' (ToolNode)
  - tools_condition routes: if the model calls tools -> ToolNode -> back to agent
                            otherwise -> END
  - MemorySaver checkpointer provides per-session conversation persistence
    keyed by thread_id (one per Gradio browser session)

Adding tools: define them in tools.py, add to ALL_TOOLS. Nothing changes here.
"""

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from tools import ALL_TOOLS

# ---------------------------------------------------------------------------
# CONFIG

MODEL = "gemma4:31b-cloud"   # swap to gemma4:31b-cloud etc. as needed

SYSTEM_PROMPT = SystemMessage(content="""\
You are a clinical pharmacist assistant powered by FDA-approved drug labeling.

You have the following tools available:
- query_drug_label: answer any clinical question from indexed drug labels
- list_indexed_drugs: show which drugs are in the database
- add_drug: download and index a new drug label on demand
- get_drug_sections: show the structure/sections of an indexed label

Rules:
- Always use query_drug_label for clinical questions (side effects, dosage, warnings, etc.)
- Use list_indexed_drugs when the user asks what's available
- Use add_drug when the user asks to add a drug not yet in the database
- Use get_drug_sections to show label structure when the user asks about it
- Never fabricate drug information — only report what the tools return
- Be concise and precise; this is medical reference information
""")

# ---------------------------------------------------------------------------
# LLM

llm = ChatOllama(model=MODEL, temperature=0.1)
llm_with_tools = llm.bind_tools(ALL_TOOLS)

# ---------------------------------------------------------------------------
# GRAPH NODES

def call_agent(state: MessagesState) -> dict:
    """Prepend the system prompt and call the LLM."""
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(ALL_TOOLS)

# ---------------------------------------------------------------------------
# GRAPH DEFINITION

def build_graph() -> StateGraph:
    graph = StateGraph(MessagesState)

    graph.add_node("agent", call_agent)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")

    # tools_condition: routes to "tools" if the last message has tool_calls,
    # otherwise routes to END
    graph.add_conditional_edges("agent", tools_condition)

    # After tools run, always return to the agent for a final response
    graph.add_edge("tools", "agent")

    return graph


# ---------------------------------------------------------------------------
# COMPILED GRAPH  (module-level singleton — one graph, many sessions)
#
# MemorySaver stores message history in memory, keyed by thread_id.
# Each Gradio session passes its own thread_id in the config dict, so
# conversation history is fully isolated between users.

memory = MemorySaver()
graph  = build_graph().compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# SESSION HELPER  — thin wrapper used by fda_app.py

class AgentSession:
    """
    Thin wrapper around the compiled LangGraph graph for a single Gradio session.
    Holds a stable thread_id so MemorySaver can restore history across turns.
    """

    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.config    = {"configurable": {"thread_id": thread_id}}

    def chat(self, user_message: str) -> str:
        """Send one user turn, return the final assistant text."""
        from langchain_core.messages import HumanMessage

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=self.config,
        )
        # The last message in state is the final AI response
        return result["messages"][-1].content

    def reset(self) -> None:
        """
        Clear this session's history by pointing to a new thread_id.
        (MemorySaver keeps the old thread; we simply stop using it.)
        """
        import uuid
        self.thread_id = str(uuid.uuid4())
        self.config    = {"configurable": {"thread_id": self.thread_id}}
