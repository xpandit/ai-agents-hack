# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Challenge 6: Agentic RAG with OpenAI Agents SDK and Azure OpenAI
#
# In this notebook, we'll explore how to use the OpenAI Agents SDK with Azure OpenAI Service to build an intelligent RAG system. This approach complements the previous Semantic Kernel example by showing an alternative implementation using OpenAI's dedicated agents framework.
#
# ## What is the OpenAI Agents SDK?
#
# The OpenAI Agents SDK is a Python library designed to help developers build agentic AI applications. It provides a simple yet powerful framework with a small set of primitives:
#
# - **Agents**: LLMs equipped with instructions and tools
# - **Handoffs**: Allow agents to delegate to other agents for specific tasks
# - **Guardrails**: Enable input validation for agents
#
# The SDK also includes built-in tracing functionality that helps visualize and debug agent workflows.

# %% [markdown]
# ## 1. Setting up Our Environment
#
# First, let's install the necessary packages.

# %%
# !pip install openai openai-agents python-dotenv azure-search-documents PyPDF2 modelcontextprotocol

# %% [markdown]
# ## 2. Initializing Azure Services
#
# Now let's import the necessary libraries and set up our connections to Azure OpenAI and Azure AI Search.

# %%
import os
import uuid
import asyncio
import json
import PyPDF2
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, 
    SimpleField, 
    SearchFieldDataType, 
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    VectorSearchAlgorithmKind
)

from openai import AsyncAzureOpenAI
from agents import Agent, Runner, function_tool, set_default_openai_client, handoff, trace, add_trace_processor
from agents.tracing.processors import ConsoleSpanExporter, BatchTraceProcessor
from modelcontextprotocol import MCPServerStdio, MCPTool

from IPython.display import display, HTML, Markdown

# Load environment variables
load_dotenv()

# Azure AI Search setup
search_service_name = os.getenv("AZURE_SEARCH_SERVICE_NAME")
search_admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
search_endpoint = f"https://{search_service_name}.search.windows.net"

# Azure OpenAI setup
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# %% [markdown]
# ## 3. Configuring Azure OpenAI with the OpenAI Agents SDK
#
# The OpenAI Agents SDK can connect to Azure OpenAI Service, allowing you to leverage your Azure deployments while using the Agents SDK functionality.

# %%
# Create Azure OpenAI clients for the main model and embeddings
openai_client = AsyncAzureOpenAI(
    api_key=azure_openai_key,
    api_version=azure_openai_api_version,
    azure_endpoint=azure_openai_endpoint,
    azure_deployment=azure_deployment
)

embedding_client = AsyncAzureOpenAI(
    api_key=azure_openai_key,
    api_version=azure_openai_api_version,
    azure_endpoint=azure_openai_endpoint,
    azure_deployment=azure_embedding_deployment
)

# Set the default OpenAI client for the Agents SDK
set_default_openai_client(openai_client)

# Set up console tracing for debugging
console_exporter = ConsoleSpanExporter()
console_processor = BatchTraceProcessor(exporter=console_exporter)
add_trace_processor(console_processor)

# %% [markdown]
# ## 4. Setting up Azure AI Search Index
#
# Let's set up our search index for storing and retrieving HR documentation.

# %%
# Define constants for index
INDEX_NAME = "hr-documents-agents"
VECTOR_DIMENSIONS = 1536  # Dimensions for text-embedding-ada-002

# Define the schema for our search index
def create_search_index(index_name: str, index_client: SearchIndexClient):
    """Create a search index if it doesn't exist."""
    
    if index_name in [index.name for index in index_client.list_indexes()]:
        print(f"Index '{index_name}' already exists")
        return
    
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SimpleField(name="title", type=SearchFieldDataType.String),
        SimpleField(name="category", type=SearchFieldDataType.String),
        SimpleField(name="page_num", type=SearchFieldDataType.Int32),
        SearchField(
            name="vector", 
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name="vector-profile"
        )
    ]
    
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="vector-algorithm", 
                kind=VectorSearchAlgorithmKind.HNSW
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="vector-profile", 
                algorithm_configuration_name="vector-algorithm"
            )
        ]
    )
    
    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    index_client.create_index(index)
    print(f"Created index '{index_name}' with vector search capability")

# Initialize search clients
search_index_client = SearchIndexClient(
    endpoint=search_endpoint,
    credential=AzureKeyCredential(search_admin_key)
)

search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(search_admin_key)
)

# Create search index
create_search_index(INDEX_NAME, search_index_client)

# %% [markdown]
# ## 5. Processing Documents and Generating Embeddings
#
# Now we'll process the employee handbook PDF and index it in Azure AI Search.

# %%
# Functions for document processing
async def generate_embeddings(text):
    """Generate embeddings for a text using Azure OpenAI."""
    try:
        response = await embedding_client.embeddings.create(
            input=text,
            model=azure_embedding_deployment
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return [0.0] * VECTOR_DIMENSIONS

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file, returning the text content by page."""
    
    print(f"Extracting text from {pdf_path}...")
    
    pdf_pages = []
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    pdf_pages.append({
                        "page_num": page_num + 1,
                        "content": text.strip()
                    })
            
            print(f"Successfully extracted text from {len(pdf_pages)} pages")
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    
    return pdf_pages

async def chunk_text_with_embeddings(pages, max_chunk_size=4000):
    """Split page content into smaller chunks with embeddings."""
    
    chunks = []
    
    for page in pages:
        page_text = page["content"]
        page_num = page["page_num"]
        
        # If the page text is shorter than max_chunk_size, keep it as is
        if len(page_text) <= max_chunk_size:
            embedding = await generate_embeddings(page_text)
            
            chunks.append({
                "page_num": page_num,
                "content": page_text,
                "title": f"Employee Handbook - Page {page_num}",
                "category": "Handbook",
                "vector": embedding
            })
        else:
            # Split by paragraphs first
            paragraphs = page_text.split('\n\n')
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) <= max_chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    # Add the current chunk if it's not empty
                    if current_chunk:
                        embedding = await generate_embeddings(current_chunk.strip())
                        
                        chunks.append({
                            "page_num": page_num,
                            "content": current_chunk.strip(),
                            "title": f"Employee Handbook - Page {page_num}",
                            "category": "Handbook",
                            "vector": embedding
                        })
                    
                    current_chunk = para + "\n\n"
            
            # Add the last chunk if it's not empty
            if current_chunk:
                embedding = await generate_embeddings(current_chunk.strip())
                
                chunks.append({
                    "page_num": page_num,
                    "content": current_chunk.strip(),
                    "title": f"Employee Handbook - Page {page_num}",
                    "category": "Handbook",
                    "vector": embedding
                })
    
    print(f"Created {len(chunks)} chunks with embeddings from {len(pages)} pages")
    return chunks

def index_documents(documents, search_client):
    """Index a list of documents into Azure AI Search."""
    
    indexed_docs = []
    
    for doc in documents:
        # Create a unique ID for each document
        doc_id = str(uuid.uuid4())
        
        # Format the document for indexing
        search_doc = {
            "id": doc_id,
            "title": doc["title"],
            "content": doc["content"],
            "category": doc["category"],
            "page_num": doc["page_num"],
            "vector": doc["vector"]
        }
        
        indexed_docs.append(search_doc)
    
    # Index the documents in batches
    search_client.upload_documents(documents=indexed_docs)
    print(f"Indexed {len(indexed_docs)} documents with vector embeddings")
    
    return indexed_docs

# Process and index the PDF
async def process_and_index_pdf():
    pdf_path = "docs/contoso_electronics.pdf"
    pdf_pages = extract_text_from_pdf(pdf_path)
    pdf_chunks = await chunk_text_with_embeddings(pdf_pages)
    
    # Index the processed chunks
    indexed_documents = index_documents(pdf_chunks, search_client)
    return indexed_documents

# Initialize the vector search
async def init_vector_search():
    # Process and index the PDF
    await process_and_index_pdf()
    print("Vector search initialized!")
    return True

# %%
# This cell initializes the vector search when executed
async def initialize():
    await init_vector_search()
    
# When running in Jupyter, uncomment and run this line:
# await initialize()

# %% [markdown]
# ## 6. Creating Function Tools for Document Search
#
# With the OpenAI Agents SDK, we can wrap search functionality as a tool that our agents can use.

# %%
# Create a function tool for searching documents
@function_tool
async def search_hr_documents(
    query: str, 
    top: int = 3
) -> str:
    """
    Search for information in the HR documentation using the provided query.
    
    Args:
        query: The search query to find relevant HR documentation
        top: The number of top results to return (default: 3)
    
    Returns:
        A formatted string with the search results
    """
    try:
        # Generate embedding for the query
        query_embedding = await generate_embeddings(query)
        
        # Perform vector search
        vector_results = search_client.search(
            search_text=None,
            vector=query_embedding,
            vector_fields="vector",
            top=top,
            select=["id", "title", "content", "page_num"]
        )
        
        # Format the results
        formatted_results = []
        
        for i, result in enumerate(vector_results):
            formatted_results.append(f"Result {i+1} (Page {result['page_num']}):\n{result['content']}\n")
        
        if formatted_results:
            return "\n".join(formatted_results)
        else:
            return "No relevant HR documents found for your query."
    
    except Exception as e:
        return f"An error occurred while searching: {str(e)}"

@function_tool
async def generate_query_from_question(question: str) -> str:
    """
    Generate an optimized search query based on the user's question.
    
    Args:
        question: The user's question about HR policies or information
    
    Returns:
        An optimized search query for better retrieval
    """
    # This will be handled by the LLM's built-in reasoning since it's a simple text transform
    return question

# %% [markdown]
# ## 7. Creating Specialized Agents with Handoffs
#
# Let's create a system of agents for our HR application, including a general assistant, a search specialist, and a policy expert.

# %%
# Create a search specialist agent
search_specialist = Agent(
    name="HR Search Specialist",
    instructions="""You are a search specialist for HR documentation. 
    Your job is to:
    1. Take a query about HR policies or employee information
    2. If needed, optimize the query using the generate_query_from_question tool
    3. Search the knowledge base using the search_hr_documents tool
    4. Return the most relevant information found
    
    Be thorough and precise in your searches. Focus on finding exact answers.
    Always cite the page number where information was found.""",
    tools=[search_hr_documents, generate_query_from_question]
)

# Create a policy expert agent
policy_expert = Agent(
    name="HR Policy Expert",
    instructions="""You are an HR policy expert. 
    Your job is to:
    1. Analyze HR policy questions
    2. Interpret and explain HR policies in a clear, helpful way
    3. Provide well-reasoned guidance on policy implementation
    4. Make sure your explanations align with company policies
    
    Focus on explaining "why" policies exist and how they benefit both the employee and the company.
    If you need to search for specific policy details, hand off to the HR Search Specialist.""",
    handoffs=[handoff(search_specialist)]
)

# Create a general HR assistant agent that can delegate to specialists
hr_assistant = Agent(
    name="HR Assistant",
    instructions="""You are a helpful HR assistant for Contoso Electronics.
    
    Help employees with HR-related questions, focusing on:
    - Company policies and procedures
    - Benefits and time off
    - Workplace guidelines
    - Onboarding information
    
    For policy interpretation or complex policy questions, hand off to the HR Policy Expert.
    For detailed information search, hand off to the HR Search Specialist.
    
    Be friendly, professional, and concise. Focus on providing accurate information based on company documentation.""",
    handoffs=[
        handoff(search_specialist),
        handoff(policy_expert)
    ]
)

# %% [markdown]
# ## 8. Testing Our Agentic RAG System with the OpenAI Agents SDK
#
# Now let's test our agents with some sample questions.

# %%
# Function to run an agent and display the result
async def ask_hr_assistant(question):
    print(f"Question: {question}\n")
    
    # Use tracing to capture the agent workflow
    with trace(workflow_name="HR Assistant RAG"):
        result = await Runner.run(
            hr_assistant,
            input=question
        )
    
    print(f"Answer: {result.response.content}\n")
    return result

# %%
# This cell can be used to test the HR assistant
async def test_hr_assistant():
    # Test with a simple question
    await ask_hr_assistant("What is the policy on remote work?")
    
    # Test with a question that might require policy interpretation
    await ask_hr_assistant("I'm planning to take parental leave soon. What is the process and how much time am I entitled to?")
    
    # Test with a question that might require specific document search
    await ask_hr_assistant("What are the security protocols for accessing company systems remotely?")

# When running in Jupyter, uncomment and run this line:
# await test_hr_assistant()

# %% [markdown]
# ## 9. Integrating MCP Bing Search Server
#
# Now let's integrate with the existing MCP server for Bing Search to add web search capabilities to our agent system.

# %%
# Set up the MCP server connection to the Bing Search service
async def setup_bing_search():
    # Connect to the MCP server
    mcp_server = MCPServerStdio(
        params={
            "command": "uv",
            "args": ["run", "-m", "mcp_server_bing_search.start"]
        }
    )
    
    # Start the server
    await mcp_server.start()
    
    # List the tools provided by the server
    tools = await mcp_server.list_tools()
    print(f"Available MCP tools: {[tool.name for tool in tools]}")
    
    return mcp_server, tools

# Function to convert MCP tools to OpenAI Agents SDK tools
def mcp_tool_to_agents_tool(mcp_tool: MCPTool):
    """Convert an MCP tool to an OpenAI Agents SDK compatible tool."""
    from agents.function_definition import FunctionDefinition
    from agents.function_tool import FunctionTool
    
    # Create a function that will call the MCP tool
    async def tool_wrapper(*args, **kwargs):
        result = await mcp_tool.call(**kwargs)
        return result
    
    # Create a function definition based on the MCP tool schema
    func_def = FunctionDefinition(
        name=mcp_tool.name,
        description=mcp_tool.description,
        parameters=mcp_tool.parameters,
        returns={"type": "string"}
    )
    
    # Create and return a FunctionTool
    return FunctionTool(func_def, tool_wrapper)

# %%
# This cell sets up the MCP Bing Search integration and creates agents with web search capability
async def create_web_enhanced_agent():
    # Set up MCP Bing Search
    mcp_server, mcp_tools = await setup_bing_search()
    
    # Convert MCP tools to OpenAI Agents SDK tools
    agents_tools = [mcp_tool_to_agents_tool(tool) for tool in mcp_tools]
    
    # Create a web search specialist agent with MCP tools
    web_search_specialist = Agent(
        name="Web Search Specialist",
        instructions="""You are a specialist in finding information on the web.
        Your job is to:
        1. Take a query about general information not found in company documents
        2. Search the web using the search tool
        3. If needed, click on specific links found in search results using the click tool
        4. Summarize the most relevant information found
        
        Be thorough and precise in your searches. Always cite your sources.""",
        tools=agents_tools
    )
    
    # Create an enhanced HR assistant with web search capabilities
    enhanced_hr_assistant = Agent(
        name="Enhanced HR Assistant",
        instructions="""You are an enhanced HR assistant for Contoso Electronics with both internal and external data access.
        
        Help employees with HR-related questions, focusing on:
        - Company policies and procedures from internal documentation
        - Supplementary information from external sources when needed
        - Benefits and time off
        - Workplace guidelines
        
        For policy interpretation, hand off to the HR Policy Expert.
        For detailed information search in company docs, hand off to the HR Search Specialist.
        For information not found in company documents, hand off to the Web Search Specialist.
        
        Clearly distinguish between company policy (internal) and general information (external).
        Be friendly, professional, and concise.""",
        handoffs=[
            handoff(search_specialist),
            handoff(policy_expert),
            handoff(web_search_specialist)
        ]
    )
    
    return enhanced_hr_assistant, mcp_server

# %%
# Function to run the enhanced agent with web search capability
async def ask_enhanced_hr_assistant(agent, question):
    print(f"Question: {question}\n")
    
    # Use tracing to capture the agent workflow
    with trace(workflow_name="Enhanced HR Assistant"):
        result = await Runner.run(
            agent,
            input=question
        )
    
    print(f"Answer: {result.response.content}\n")
    return result

# %%
# This cell can be used to test the enhanced HR assistant with web search
async def test_web_enhanced_agent():
    # Create the enhanced agent with web search
    enhanced_agent, mcp_server = await create_web_enhanced_agent()
    
    try:
        # Test with a question that might require external information
        await ask_enhanced_hr_assistant(enhanced_agent, "How does our parental leave policy compare to typical laws in the US?")
        
        # Test with another question requiring external context
        await ask_enhanced_hr_assistant(enhanced_agent, "What are best practices for securing my home office for remote work?")
    finally:
        # Always close the MCP server connection when done
        await mcp_server.stop()

# When running in Jupyter, uncomment and run this line:
# await test_web_enhanced_agent()

# %% [markdown]
# ## Conclusion
#
# In this notebook, we've demonstrated how to use the OpenAI Agents SDK with Azure OpenAI to build an intelligent RAG system, integrating with the MCP protocol for web search capabilities. Key features we explored include:
#
# 1. **Azure OpenAI Integration** - Connected the OpenAI Agents SDK to Azure OpenAI Service
# 2. **Function Tools** - Created tools for search and query generation
# 3. **Agent Specialization** - Built specialized agents for different aspects of HR assistance
# 4. **Handoffs** - Enabled delegation between agents for better task handling
# 5. **MCP Protocol Integration** - Connected to an existing MCP server for Bing Search
# 6. **Tracing** - Used the built-in tracing for monitoring agent workflows
#
# This approach offers a powerful alternative to the Semantic Kernel implementation shown earlier, with a focus on agent specialization and handoffs for complex tasks.
