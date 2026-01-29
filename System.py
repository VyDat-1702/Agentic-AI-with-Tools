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
Role: Main agent

Instructions:
1. Always start with THOUGHT, then decide on ACTION (RETRIEVE knowledge or SEARCH on website) or ANSWER like FRIENDLY ASSISTANT.
2. Carefully check past tool_observations to see if the answer is already available.
3. If not, choose the most relevant tool to gather more information.

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


config = Config()
gemini_model = Gemini(config)

    
def call_agent(state: AgentState) -> AgentState:
    observations = '\n\n'.join(state.get('tool_observations', []))
    if not observations:
        observations = 'None yet - first turn'
    
    tools_list = build_tools_list()
    
    prompt = f"""
{AGENT_INSTRUCTION}

{tools_list}

USER QUERY: {state.get('query')}

PAST TOOL OBSERVATIONS: 
{observations}

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
        print("→ Routing to END (found ANSWER)")
        return "end"
    
    if "ACTION:" in response:
        print("→ Routing to TOOLS (found ACTION)")
        return "continue"
    
    if state.get("num_steps", 0) >= 10:
        print("→ Routing to END (max steps reached)")
        return "end"
    
    print("→ Routing to END (no action found)")
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