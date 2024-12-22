#!/usr/bin/env python
# coding: utf-8

# ## Writing Assistant - NoteWise
# 
# ### Description
# NoteWise is a writing assistant, with the following functionalities
# 
# `Topic Selection`
# * The use enter the topic they want to write about
# 
# `Template`
# * The user can select the template they want to use for their writing
# 
# `subtopic Generation`
# * The Agent generates subtopics based on the topics, reserach (websearch)
# * The user can select the subtopics they want to write about 
# 
# `Research Tools`
# * Research tools include wikipedia, web search and user input(content from the user)
# * The user can select the research tool they want to use for their writing
# 
# `Document Writing`
# * The Agent generates the writing based on the research tools and the user input
# * The user can confirm the writing or correct it 
# 

# In[1]:


# Imports
import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

load_dotenv()


# In[2]:


OPENAI_KEY = os.getenv("OPENAI_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# In[3]:


llm = ChatOpenAI(model = "gpt-4o", temperature = 0)


# In[41]:


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


# In[42]:


from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

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



# In[43]:


topic = "what is Artificial intelligence"
writing_type = "Blog"
max_subtopics = 5
template = """
Title: [Enter the blog title here]

Introduction:
[Write an engaging introduction that explains what the blog will cover and why it's important.]

Table of Contents:
1. [What is Topic?]
2. [Why is Topic Important?]
3. [Key Insights]
4. [How to Implement Topic]
5. [Challenges and Solutions]
6. [Conclusion]

Body:
1. What is Topic?
   [Explain the topic clearly with examples or analogies.]

2. Why is Topic Important?
   [Discuss the significance of the topic with supporting data or real-world examples.]

3. Key Insights:
   - [Key Insight 1]
   - [Key Insight 2]
   - [Key Insight 3]

4. How to Implement Topic:
   - Step 1: [Explain step 1]
   - Step 2: [Explain step 2]
   - Step 3: [Explain step 3]

5. Challenges and Solutions:
   - Challenge 1: [Describe challenge]
     - Solution: [Provide solution]
   - Challenge 2: [Describe challenge]
     - Solution: [Provide solution]

Conclusion:
[Summarize the blog's key points and end with a call-to-action or final thoughts.]

References/Further Reading:
- [Reference 1]
- [Reference 2]
"""

thread = {"configurable": {"thread_id": "1"}}
graph.invoke({"topic":topic, "writing_type": writing_type, "max_subtopics": max_subtopics, "template": template }, thread)


# In[44]:


state = graph.get_state(thread)
state.values


# In[45]:


graph.update_state(thread, {"human_feedback":"Dont add the history and all that, am interested in topics like the difference between ai and ml, what is gen ai"})
# graph.get_state(thread).values


# In[46]:


state = graph.get_state(thread)
state.next


# In[47]:


graph.invoke(None,thread)


# In[48]:


graph.update_state(thread, {"human_feedback":None}, as_node= "human_feedback_function")


# In[49]:


# the graph is now at the end, lets see whats next
state = graph.get_state(thread)
state.values["subtopics"][0].title


# ## Generating the actual writing and human in loop for confirming the article writing
# - sofar we have been able to generate the subtopics and get human feeback on these topics, now its time to generate the writing and allow the user to confirm

# In[ ]:


from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

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
    print("I have generated these topics", subtopics)
    return {"subtopics" : subtopics.subTopics}

def human_feedback_function(state: GenerateWritingState):
    pass

def human_feedback_writing(state: GenerateWritingState):
    pass
def subtopics_confirmation(state: GenerateWritingState):
    human_feedback = state.get("human_feedback", None)
    
    if human_feedback:
        print("am deciding loop", human_feedback)
        return "generate_subtopics"
    print("am deciding to end", human_feedback)
    return "generate_writing"

writing_prompt = """
You are an expert in writing a {writing_type}.
You are tasked with creating a {writing_type} with these subtopics {subtopics} under this topic {topic}.
1. Analyse each subtopic and generate content under the subtopic
2. your output should stick to this template example {template}
"""
def generate_writing(state: GenerateWritingState):
    print("I have started writing")
    """
    Generate writing based on the given state.
    
    Args:
    state (GenerateWritingState): The state of the writing generation process.
    
    Return:
    writing: The generated writing update to the state.
    """
    writing_type = state["writing_type"]
    subtopics = state["subtopics"]
    template = state["template"]
    topic = state["topic"]
    writing_instruction = state["writing_instruction"]
    formatted_string = f"'subtopics': [{', '.join([f'SubTopic(title={repr(subtopic.title)}, content={repr(subtopic.content)})' for subtopic in subtopics])}]"
    print("formatted string", formatted_string)
    print("before formatted writing")
    
    formatted_writing_prompt = writing_prompt.format(writing_type = writing_type, subtopics = formatted_string, template= template, topic = topic)
    print("formatted_writing_prompt", formatted_writing_prompt)
    response = llm.invoke([SystemMessage(content=formatted_writing_prompt)]+[HumanMessage(content=writing_instruction)])
    print("after calling the llm")
    
    print(response)
    return {"writing": response.content}

def writing_confirm(state: GenerateWritingState):
    writing_confirm = state['writing_confirm']
    if writing_confirm:
        return "END"
    return "generate_writing"
# create a graph builder
builder = StateGraph(GenerateWritingState)

# Add a node
builder.add_node(generate_subtopics, "generate_subtopics")
builder.add_node(human_feedback_function, "human_feedback_function")
builder.add_node(human_feedback_writing, "human_feedback_writing")
builder.add_node(generate_writing, "generate_writing")

# add edges
builder.add_edge(START, "generate_subtopics")
builder.add_edge("generate_subtopics", "human_feedback_function")
builder.add_conditional_edges("human_feedback_function", subtopics_confirmation, ["generate_subtopics", "generate_writing"])
builder.add_edge("generate_writing", "human_feedback_writing")
builder.add_conditional_edges("human_feedback_writing", "writing_confirm", ["generate_writing", "END"])


# Compile
memory = MemorySaver()
graph_2 = builder.compile(interrupt_before=['human_feedback_function', 'generate_writing'],checkpointer=memory)

# View
display(Image(graph_2.get_graph(xray=1).draw_mermaid_png()))



    


# In[ ]:


topic = "what is Artificial intelligence"
writing_type = "Blog"
max_subtopics = 3
template = """
Blog Title

Introduction
    - Brief overview of the topic
    - Hook to engage the reader

Subtopic 1: Background/Context
    - Explanation of key terms or concepts
    - Importance or relevance of the topic

Subtopic 2: Main Discussion/Analysis
    - Detailed exploration of the topic
    - Supporting arguments or evidence
    - Examples or case studies

Subtopic 3: Challenges or Considerations
    - Possible issues or complexities
    - Contrasting perspectives (if applicable)

Subtopic 4: Solutions/Insights
    - Proposed solutions or strategies
    - Key takeaways or lessons

Conclusion
    - Summary of main points
    - Call to action or closing thoughts

Optional: References/Further Reading
    - Links or citations for deeper exploration
"""

thread_2 = {"configurable": {"thread_id": "2"}}
graph_2.invoke({"topic":topic, "writing_type": writing_type, "max_subtopics": max_subtopics, "template": template }, thread_2)


# In[ ]:


graph_2.update_state(thread_2,{"human_feedback": "dont talk about the history, talk about the types of of AI and gen AI"}, as_node="human_feedback_function")


# In[ ]:


graph_2.get_state(thread_2).values


# In[ ]:


graph_2.invoke(None, thread_2)


# In[ ]:


graph_2.update_state(thread_2,{"human_feedback": None})


# In[ ]:


graph_2.get_state(thread_2).values


# In[ ]:


graph_2.invoke(None, thread_2)


# In[ ]:




