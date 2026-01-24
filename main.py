import json
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import google.generativeai as genai

from Tools import initialize_tools, TOOLS_MAPPING_2_FUNCTIONS, TOOLS_DESCRIPTION

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = "gemini-2.5-flash"
    
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./QdrantDB")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medical_qa_kb")

class GeminiModel:
    def __init__(self):
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env file!")
        
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        
        print(f"Gemini model initialized: {Config.GEMINI_MODEL}")
    
    def invoke(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                )
            )
            return response.text.strip()
            
        except Exception as e:
            return f"Gemini error: {e}"


llm = GeminiModel()

AGENT_SYSTEM_INSTRUCTION = """You are a Medical Information Assistant with tool access.

CRITICAL SAFETY RULES:
1. ALWAYS include: "This is for informational purposes only. Consult a healthcare professional."
2. NEVER diagnose or prescribe medications
3. For emergencies → advise call 115 or go to ER immediately

WORKFLOW:
1. THOUGHT: Analyze what to do
2. ACTION: get_medical_faq (try FIRST) or web_search_medical
3. Review PAST TOOL OBSERVATIONS
4. When sufficient info → provide ANSWER

FORMAT:
THOUGHT: [Your reasoning]
ACTION: [tool_name]
ARGUMENTS: {"query": "..."}

OR:

THOUGHT: [Why ready to answer]
ANSWER: [Complete response with disclaimer]

RULES:
- ARGUMENTS must be valid JSON with double quotes
- Check PAST TOOL OBSERVATIONS before calling tools again
- Use get_medical_faq FIRST (16K+ Q&A database)
- Use web_search_medical only if FAQ insufficient
- Always end ANSWER with safety disclaimer
"""

class AgentState(TypedDict):
    query: str
    last_agent_response: str
    tool_observations: list
    num_steps: int


def call_agent(state: AgentState) -> AgentState:
    observations = "\n\n".join(state.get("tool_observations", []))
    if not observations:
        observations = "None yet - first turn"
    
    prompt = f"""{AGENT_SYSTEM_INSTRUCTION}

{TOOLS_DESCRIPTION}

USER QUERY: {state['query']}

PAST TOOL OBSERVATIONS:
{observations}

What do you do next?"""
    
    response = llm.invoke(prompt)
    
    state["last_agent_response"] = response
    state["num_steps"] += 1
    
    print()
    print(f"🤖 AGENT (Step {state['num_steps']})")
    print(response)
    
    return state


def call_tools(state: AgentState) -> AgentState:
    action_text = state.get("last_agent_response", "")
    
    if "ACTION:" not in action_text:
        state.setdefault("tool_observations", []).append(
            "No ACTION found"
        )
        return state
    
    print()
    print("EXECUTING TOOL")
    
    # Extract tool name
    try:
        tool_name = action_text.split("ACTION:")[1].split("\n")[0].strip()
    except:
        state.setdefault("tool_observations", []).append("Failed to parse ACTION")
        return state
    
    args = {}
    if "ARGUMENTS:" in action_text:
        try:
            args_text = action_text.split("ARGUMENTS:")[1].strip()
            if "{" in args_text:
                json_start = args_text.index("{")
                json_end = args_text.rindex("}") + 1
                json_str = args_text[json_start:json_end]
                args = json.loads(json_str)
        except Exception as e:
            state.setdefault("tool_observations", []).append(
                f"Failed to parse ARGUMENTS: {e}"
            )
            return state
    
    # Execute
    tool_func = TOOLS_MAPPING_2_FUNCTIONS.get(tool_name)
    
    if not tool_func:
        state.setdefault("tool_observations", []).append(
            f"Unknown tool: {tool_name}"
        )
        return state
    
    try:
        print(f"Tool: {tool_name}")
        print(f"Args: {json.dumps(args, indent=2)}")
        
        results = tool_func(**args)
        
        observation = f"""[Tool: {tool_name}]
Source: {results.get('source', 'Unknown')}
Results:
{results.get('context', 'No context')}
"""
        
        state.setdefault("tool_observations", []).append(observation)
        
        print(f"Success")
        
    except Exception as e:
        state.setdefault("tool_observations", []).append(
            f"Tool error: {e}"
        )
    
    return state

def should_continue(state: AgentState) -> str:

    response = state.get("last_agent_response", "").upper()
    
    if "ANSWER:" in response:
        print("END (found ANSWER)")
        return "end"
    
    if "ACTION:" in response:
        print("TOOLS (found ACTION)")
        return "continue"
    
    if state.get("num_steps", 0) >= 10:
        print("END (max steps)")
        return "end"
    
    print("END (no action)")
    return "end"

def build_graph():

    workflow = StateGraph(state_schema=AgentState)
    
    workflow.add_node("agent", call_agent)
    workflow.add_node("tools", call_tools)
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


def run_query(query: str, graph):
    print()
    print(f"# 💬 USER: {query}")
    print()
    
    state = {
        "query": query,
        "last_agent_response": "",
        "tool_observations": [],
        "num_steps": 0
    }
    
    result = graph.invoke(state)
    
    response = result.get("last_agent_response", "")
    
    if "ANSWER:" in response:
        answer = response.split("ANSWER:", 1)[1].strip()
    else:
        answer = response
    
    print()
    print("FINAL ANSWER:")
    print()
    print(answer)
    print()
    
    return answer


def main():
    
    print()
    print("🏥 MEDICAL AGENT - GEMINI + QDRANT")
    print()
    

    print("\nInitializing...")
    initialize_tools()
    
    print("\nBuilding graph...")
    graph = build_graph()
    
    print("\nReady!")
    print()
    
    # Test queries
    queries = [
        "What are symptoms of diabetes?",
        "How to prevent heart disease?",
        "Latest COVID-19 treatment options?",
    ]
    
    for i, q in enumerate(queries, 1):
        print(f"TEST {i}/{len(queries)}")
        
        run_query(q, graph)
    
        if i < len(queries):
            input("\n Press Enter...")
    
    print("\n All tests done!")


if __name__ == "__main__":
    main()