# NoteWise

NoteWise is an intelligent agent designed to answer questions about specific subjects using a combination of tools including Wikipedia, Arxiv, and a custom vector store. This project leverages the power of various APIs and machine learning models to provide accurate and comprehensive information retrieval.

## Features

- **Wikipedia Tool**: Utilizes the Wikipedia API to fetch and summarize information from Wikipedia articles.
- **Arxiv Tool**: Accesses scholarly articles from Arxiv to provide detailed academic information.
- **Vector Store Tool**: Uses a custom-built vector store to retrieve information from a pre-loaded dataset.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/francoismavunila/NoteWise.git
    cd notewise
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Load the necessary tools:
    ```python
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.tools.retriever import create_retriever_tool
    from langchain_community.utilities import ArxivAPIWrapper
    from langchain_community.tools import ArxivQueryRun
    ```

2. Initialize the tools:
    ```python
    # Wikipedia Tool
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200))

    # Arxiv Tool
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

    # Vector Store Tool
    loader = WebBaseLoader("link_to_any_site_here")
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
    retriever = vectordb.as_retriever()
    retriever_tool = create_retriever_tool(retriever, "too_name", "search for information about ......, for any information about ...... use this tool")

    # Combine tools
    tools = [wikipedia, arxiv, retriever_tool]
    ```

3. Use the tools to answer questions:
    ```python
    # Example usage
    question = "What is the latest research on AI?"
    response = agent.answer(question, tools=tools)
    print(response)
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.