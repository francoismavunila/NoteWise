import gradio as gr
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
from fpdf import FPDF

load_dotenv()

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
    formatted_string = f"'subtopics': [{', '.join([f'SubTopic(title={repr(subtopic.title)}, content={repr(subtopic.content)})' for subtopic in subtopics])}]"
    print("formatted string", formatted_string)
    print("before formatted writing")
    
    formatted_writing_prompt = writing_prompt.format(writing_type = writing_type, subtopics = formatted_string, template= template, topic = topic)
    print("formatted_writing_prompt", formatted_writing_prompt)
    response = llm.invoke([SystemMessage(content=formatted_writing_prompt)]+[HumanMessage(content="generate writing")])
    print("after calling the llm")
    
    print(response)
    return {"writing": response.content}
def writing_confirm(state: GenerateWritingState):
    writing_confirm = state.get("writing_confirm", None)
    if writing_confirm:
        return "generate_writing"
    return END

# create a graph builder
builder = StateGraph(GenerateWritingState)

# Add a node
builder.add_node(generate_subtopics, "generate_subtopics")
builder.add_node(human_feedback_function, "human_feedback_function")
builder.add_node(generate_writing, "generate_writing")
builder.add_node(human_feedback_writing, "human_feedback_writing")

# add edges
builder.add_edge(START, "generate_subtopics")
builder.add_edge("generate_subtopics", "human_feedback_function")
builder.add_conditional_edges("human_feedback_function", subtopics_confirmation, ["generate_subtopics", "generate_writing"])
builder.add_edge("generate_writing", "human_feedback_writing")
builder.add_conditional_edges("human_feedback_writing", writing_confirm, ["generate_writing", END])


# Compile
memory = MemorySaver()
graph = builder.compile(interrupt_before=['human_feedback_function', 'human_feedback_writing'],checkpointer=memory)


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

thread = {"configurable": {"thread_id": "1"}}


# Function to save the writing as a PDF
def save_as_pdf(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in content.split('\n'):
        pdf.multi_cell(0, 10, line)
    file_path = "writing_output.pdf"
    pdf.output(file_path)
    return file_path

# Function to handle the "Finish" action
def finish_writing():
    # Get the final writing content from the graph state
    final_state = graph.get_state(thread)
    writing_content = final_state.values.get("writing", "")
    
    # Generate the file path for the saved PDF
    pdf_file_path = save_as_pdf(writing_content)
    
    # Return the writing content and the PDF file for download
    return (
        gr.update(visible = False),
        gr.update(visible = False),
        gr.update(visible = False),
        gr.update(value=writing_content, visible=True),
        pdf_file_path,
        gr.update(visible=True)
    )
    
# now dealing with gradio interface 
def regenerate_subtopics(feedback):
    # print(graph.get_state(thread).values)
    graph.update_state(thread, {"human_feedback": feedback})
    subtopics = graph.invoke(None, thread)
    return format_subtopics(subtopics["subtopics"])

def re_write(feedback):
    graph.update_state(thread, {"writing_confirm": feedback})
    answer = graph.invoke(None, thread)
    return answer["writing"]

def generate_writing():
    graph.update_state(thread, {"human_feedback": None})
    answer = graph.invoke(None, thread)
    print("the state after writing",answer)
    return answer["writing"], gr.update(visible = False), gr.update(visible = False), gr.update(visible = True),gr.update(visible = True), gr.update(value="")

def format_subtopics(subtopic_list):
    subtopics_str = "\n\n".join([f"**Subtopic {i+1}:** {subtopic.title}" for i, subtopic in enumerate(subtopic_list)])
    return f"""
## Here are your subtopics:
{subtopics_str}

---
### Please give feedback on these topics or Click Continue to generate your writing.
    """
with gr.Blocks() as demo:
    # first screen
    with gr.Row(visible=True) as sub_topics_row:
        title = gr.Textbox(label="Name")
        num_sub_topics = gr.Number(label="Number of subtopics", value = 4)
        writing_type = gr.Dropdown(choices=["blog", "article"], label="Type")
        sub_topic_btn = gr.Button("Get subtopic")
    output_area = gr.Markdown(visible = False)
    feedback_input = gr.Textbox(label="feedback", visible=False)
    feedback_btn = gr.Button("Re-generate", visible=False)
    continue_btn = gr.Button("Continue", visible=False)
    
    re_write_btn = gr.Button("Re-write", visible = False)
    finish_btn = gr.Button("Finish", visible = False)
    pdf_download = gr.File(label="Download PDF", visible=False)
    
    
    def get_subtopics(title, num_sub_topics, writing_type):
        answer = graph.invoke({"topic":title, "writing_type": writing_type, "max_subtopics": num_sub_topics, "template": template, "writing_instruction": "generate writing" }, thread)
        subtopics_str = format_subtopics(answer["subtopics"])
        return gr.update(visible=False), gr.update(value=subtopics_str, visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    sub_topic_btn.click(fn=get_subtopics, inputs=[title, num_sub_topics, writing_type], outputs=[sub_topics_row, output_area, feedback_input, feedback_btn, continue_btn])
    feedback_btn.click(
        fn=regenerate_subtopics, 
        inputs=feedback_input, 
        outputs=output_area
    )
    continue_btn.click(
        fn = generate_writing,
        inputs = None,
        outputs = [output_area, feedback_btn, continue_btn, re_write_btn, finish_btn, feedback_input]
    )
    re_write_btn.click(
        fn = re_write,
        inputs = feedback_input,
        outputs = output_area
    )
    finish_btn.click(
        fn=finish_writing,
        inputs=None,
        outputs=[feedback_input,re_write_btn,finish_btn,output_area, pdf_download, pdf_download]
    )
    
        
        

demo.launch()