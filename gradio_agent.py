import gradio_test as gr
import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# get secret keys
OPENAI_KEY = os.getenv("OPENAI_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

llm = ChatOpenAI(model = "gpt-4o", temperature = 0)

## define states
class SubTopic(BaseModel):
    title: str = Field(
        description= "title for the subtopic"
    )
    content: str = Field(
        description="topic for the subtopic"
    )

class SubTopicList(BaseModel):
    subTopics: List[SubTopic] = Field(
        description= "a comprehensive list of subtopics"
    )
    
class GenerateWritingState(TypedDict):
    topic: str
    writing_type:str
    subtopics: List[SubTopic]
    max_subtopics: int
    human_feedback: str
    template: str
    tools: List[str]
    writing: str
    writing_confirm: bool = False
    writing_instruction: str = "generate writing"
    
generate_subtopics_prompt = """
You are tasked with creating subtopics for a {writing_type}, follow these instructions carefully:
1. Carefully analyse this topic to understand what it is and what is required of it:
{topic}
2. Examine any editorial feedback that has been optionally provided to guide in the creation of subtopics
{human_feedback}
3. Dertemine the most interesting sub topics and pick the top {max_subtopics}
4. In the content part put an empty string for now
"""
def generate_subtopics(state: GenerateWritingState):
    
    writing_type = state["writing_type"]
    topic = state["topic"]
    human_feedback = state.get("human_feedback", None)
    max_subtopics = state["max_subtopics"]
    
    formatted_prompt = generate_subtopics_prompt.format(writing_type = writing_type, topic = topic, human_feedback = human_feedback, max_subtopics = max_subtopics)
    
    structured_llm = llm.with_structured_output(SubTopicList)
    
    subtopics = structured_llm.invoke([SystemMessage(content=formatted_prompt)] + [HumanMessage(content="generate subtopics")])
    
    return {"subtopics" : subtopics.subTopics}

def human_feedback_function(state: GenerateWritingState):
    pass

def subtopics_confirmation(state: GenerateWritingState):
    human_feedback = state.get("human_feedback", None)
    
    if human_feedback:
        print("am deciding loop", human_feedback)
        return "generate_subtopics"
    print("am deciding to end", human_feedback)
    return END

# create a graph builder
builder = StateGraph(GenerateWritingState)

# Add a node
builder.add_node(generate_subtopics, "generate_subtopics")
builder.add_node(human_feedback_function, "human_feedback_function")

# add edges
builder.add_edge(START, "generate_subtopics")
builder.add_edge("generate_subtopics", "human_feedback_function")
builder.add_conditional_edges("human_feedback_function", subtopics_confirmation, ["generate_subtopics", END])

# Compile
memory = MemorySaver()
graph = builder.compile(interrupt_before=['human_feedback_function'],checkpointer=memory)

# View
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))