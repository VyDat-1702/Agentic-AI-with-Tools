import os
import torch
import google.generativeai as genai
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict
import json
from Tools import TOOLS_MAPPING_TO_FUNC, AGENT_TOOLS_LIST
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class Config:
    GEMINI_API = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = 'gemini-2.5-flash'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
class Gemini:
    def __init__(self, config):
        self.config = config
        genai.configure(api_key=self.config.GEMINI_API)
        self.llm = genai.GenerativeModel(self.config.GEMINI_MODEL)
        
    def invoke(self, prompt: str) -> str:
        try:
            response = self.llm.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=1024,
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f'Gemini ERROR: {str(e)}')
            return ""
    
    
def build_tools_list() -> str:
    tools = AGENT_TOOLS_LIST.get('TOOLS', [])
    tool_line = ['Available tools:']
    for i, tool in enumerate(tools, 1):
        tool_line.append(
            f"[{i}]: {tool['name']}\n"
            f"    Description: {tool['description']}\n"
            f"    Arguments: {tool['args']}"
        )
    return '\n'.join(tool_line)


AGENT_INSTRUCTION = """
Role: Medical Information Assistant

CRITICAL RULES:
1. FIRST, check if PAST TOOL OBSERVATIONS already contains the answer
2. If observations have relevant information → PROVIDE ANSWER IMMEDIATELY
3. Only call tools if you truly need MORE information
4. NEVER call the same tool twice with similar queries

WORKFLOW:
Step 1: Read PAST TOOL OBSERVATIONS carefully
Step 2: Decide:
   - Has answer? → Write ANSWER
   - Need info? → Call get_qa_retriever (try FIRST)
   - FAQ failed? → Call get_web_search

OUTPUT FORMAT:

If you have enough information:
THOUGHT: [Explain why you can answer based on past observations]
ANSWER: [Your detailed answer with medical disclaimer]

If you need to use a tool:
THOUGHT: [Explain what information is missing]
ACTION: [get_qa_retriever or get_web_search]
ARGUMENTS: {"query": "specific search query"}

MANDATORY RULES:
- ARGUMENTS must be valid JSON with double quotes
- If PAST TOOL OBSERVATIONS contains relevant results → ANSWER NOW, don't call tools again
- Each tool should only be called ONCE per query unless you get an error
"""


class AgentState(TypedDict):
    query: str
    last_agent_response: str
    tool_observations: list
    num_steps: int 


config = Config()
gemini_model = Gemini(config)

    
def call_agent(state: AgentState) -> AgentState:
    observations = '\n\n'.join(state.get('tool_observations', []))
    if not observations:
        observations = 'None yet - first turn'
    
    tools_list = build_tools_list()
    
    # Count how many times each tool was called
    tool_calls_count = {}
    for obs in state.get('tool_observations', []):
        if 'TOOL:' in obs:
            tool_name = obs.split('TOOL:')[1].split('\n')[0].strip()
            tool_calls_count[tool_name] = tool_calls_count.get(tool_name, 0) + 1
    
    tools_used = ', '.join([f"{k}: {v}x" for k, v in tool_calls_count.items()]) if tool_calls_count else "None"
    
    prompt = f"""
{AGENT_INSTRUCTION}

{tools_list}

USER QUERY: {state.get('query')}

TOOLS ALREADY USED: {tools_used}

PAST TOOL OBSERVATIONS: 
{observations}

 IMPORTANT: If the observations above contain relevant information, ANSWER NOW. Do not call tools again unless absolutely necessary.

Respond now:
"""
    
    response = gemini_model.invoke(prompt=prompt)
    state['last_agent_response'] = response
    state['num_steps'] = state.get('num_steps', 0) + 1
    
    print(f'\n=== AGENT STEP {state["num_steps"]} ===')
    print(response)
    print('='*50)
    
    return state


def call_tool(state: AgentState) -> AgentState:
    action_text = state.get('last_agent_response', '')
    
    if 'ACTION:' not in action_text:
        state.setdefault('tool_observations', []).append('No ACTION found')
        return state
    
    try:
        # Extract tool name
        tool_name = None
        for line in action_text.split('\n'):
            if line.strip().startswith('ACTION:'):
                tool_name = line.split('ACTION:')[1].strip()
                break
        
        if not tool_name:
            state.setdefault('tool_observations', []).append('Could not extract tool name')
            return state
        
        # Extract arguments
        arguments = {}
        for line in action_text.split('\n'):
            if line.strip().startswith('ARGUMENTS:'):
                args_str = line.split('ARGUMENTS:')[1].strip()
                arguments = json.loads(args_str)
                break
        
        # Get tool function
        tool_func = TOOLS_MAPPING_TO_FUNC.get(tool_name)
        
        if not tool_func:
            state.setdefault('tool_observations', []).append(f'Tool {tool_name} not found')
            return state
        
        # Execute tool
        print(f'\n>>> Executing tool: {tool_name} with args: {arguments}')
        result = tool_func(**arguments)
        
        observation = f'TOOL: {tool_name}\nRESULT: {result}'
        state.setdefault('tool_observations', []).append(observation)
        
        logger.info(f'Tool {tool_name} executed successfully')
        
    except json.JSONDecodeError as e:
        state.setdefault('tool_observations', []).append(f'JSON parsing error: {str(e)}')
        logger.error(f'JSON error: {e}')
    except Exception as e:
        state.setdefault('tool_observations', []).append(f'Tool execution error: {str(e)}')
        logger.error(f'Tool execution error: {e}')
    
    return state
    
    
def should_continue(state: AgentState) -> str:
    response = state.get("last_agent_response", "").upper()
    
    if "ANSWER:" in response:
        print("Routing to END (found ANSWER)")
        return "end"
    
    if "ACTION:" in response:
        print("Routing to TOOLS (found ACTION)")
        return "continue"
    
    if state.get("num_steps", 0) >= 5:
        print("→ Routing to END (max steps reached)")
        return "end"
    
    print("Routing to END (no action found)")
    return "end"


def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node('agent', call_agent)
    workflow.add_node('tools', call_tool)
    
    workflow.set_entry_point('agent')
    
    workflow.add_conditional_edges(
        'agent',
        should_continue,
        {
            'continue': 'tools',
            'end': END
        }
    )
    
    workflow.add_edge('tools', 'agent')
    
    return workflow.compile()


def run_query(query: str, graph) -> str:
    state = {
        "query": query,
        "last_agent_response": "",
        "tool_observations": [],
        "num_steps": 0
    }
    
    result = graph.invoke(state)
    response = result.get('last_agent_response', '')
    
    if 'ANSWER:' in response:
        answer = response.split('ANSWER:', 1)[1].strip()
    else:
        answer = response
        
    return answer


def main():
    print("Initializing Medical Chatbot...")
    graph = build_graph()
    print("Ready! Type 'quit', 'exit', or 'esc' to stop.\n")
    
    while True:
        query = input('User: ').strip()
        if query.lower() in ['quit', 'exit', 'esc']:
            print("Goodbye!")
            break
        
        if not query:
            continue
            
        try:
            response = run_query(query=query, graph=graph)
            print(f'\nBot: {response}')
            print('---' * 20 + '\n')
        except Exception as e:
            logger.error(f'Error processing query: {e}')
            print(f"Sorry, an error occurred: {e}\n")
        
        
if __name__ == '__main__':
    main()