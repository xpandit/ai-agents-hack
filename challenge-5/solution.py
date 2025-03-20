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
# # Challenge 5: Tool Usage & Agentic RAG
#
# In this challenge, we'll explore how to build an intelligent assistant that leverages Retrieval-Augmented Generation (RAG) in an agentic context. We'll create a system that can index company documentation, retrieve relevant information, and intelligently answer user queries.
#
# ## What is Agentic RAG?
#
# Traditional RAG systems follow a linear process: retrieve relevant documents → generate a response based on those documents. Agentic RAG takes this further by adding intelligent decision-making to the retrieval process:
#
# - **Dynamic Query Formulation**: The agent reformulates queries to improve search results
# - **Selective Retrieval**: The agent decides when to retrieve information and when to rely on its own knowledge
# - **Multi-step Reasoning**: The agent can perform multiple retrieval steps for complex questions
# - **Tool Integration**: The agent combines retrieval with other capabilities (calculations, API calls, etc.)
#
# ## The Knowledge Base and Azure AI Search
#
# A knowledge base is a specialized database designed to store, organize, and retrieve information. In the context of AI applications:
#
# - **Knowledge bases** store structured or unstructured content (documents, FAQs, policies, etc.)
# - They're organized to facilitate quick and accurate information retrieval
# - They serve as the "memory" for AI agents, extending their knowledge beyond training data
#
# **Azure AI Search** (formerly Azure Cognitive Search) is Microsoft's cloud search service that enables:
#
# - **Document Ingestion**: Processing various file types (PDFs, Word, HTML, images with OCR, etc.)
# - **Indexing**: Creating searchable indexes with text analysis capabilities
# - **Semantic Search**: Using AI to understand query intent and contextual meaning
# - **Vector Search**: Utilizing embeddings to find conceptually similar content
# - **Hybrid Approaches**: Combining keyword and semantic search for optimal results
#
# In our agentic RAG system, Azure AI Search serves as the foundation for our knowledge base, enabling intelligent information retrieval to power our HR assistant.

# %% [markdown]
# ## 1. Setting up Our Environment
#
# First, let's install the necessary packages for our Agentic RAG implementation.

# %%
# !pip install openai python-dotenv azure-search-documents semantic-kernel azure-identity PyPDF2

# %%
import os
import json
import uuid
import PyPDF2
from dotenv import load_dotenv
from typing import List, Dict, Any, Annotated, Optional

import asyncio
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

from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.agents import ChatCompletionAgent

from IPython.display import display, HTML, Markdown

# Load environment variables
load_dotenv()

# %% [markdown]
# ## 2. Initializing Azure Services
#
# Now let's set up our connections to Azure AI Search and Azure OpenAI.

# %%
# Azure AI Search setup
search_service_name = os.getenv("AZURE_SEARCH_SERVICE_NAME")
search_admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
search_endpoint = f"https://{search_service_name}.search.windows.net"

# Azure OpenAI setup
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")  # Update to match your endpoint

# Initialize the asynchronous OpenAI client with proper Azure configuration
client = AsyncAzureOpenAI(
    azure_endpoint=azure_openai_endpoint,
    azure_deployment=azure_deployment,
    api_key=azure_openai_key,
    api_version = azure_openai_api_version
)

embedding_client = AsyncAzureOpenAI(
    azure_endpoint=azure_openai_endpoint,
    azure_deployment=azure_embedding_deployment,
    api_key=azure_openai_key,
    api_version = azure_openai_api_version
)

# Create a Semantic Kernel instance
kernel = Kernel()
chat_completion_service = OpenAIChatCompletion(
    ai_model_id=azure_deployment,
    async_client=client,
    service_id="agent",
)
kernel.add_service(chat_completion_service)

# %% [markdown]
# ## Knowledge Base Creation with Azure AI Search
#
# Let's explore in more detail how Azure AI Search functions as a knowledge base for our system:
#
# ### Key Components of an Azure AI Search Knowledge Base
#
# 1. **Data Source Connection**: 
#    - Azure AI Search connects to various data sources, including blob storage, SQL databases, and Cosmos DB
#    - In our example, we directly parsed a PDF document into text
#
# 2. **Indexing Pipeline**:
#    - **Extraction**: Converting documents into text (e.g., extracting from PDFs)
#    - **Chunking**: Breaking down documents into smaller, manageable pieces
#    - **Enrichment**: Adding metadata, entity extraction, or image analysis
#    - **Normalization**: Transforming text for better search (lowercasing, lemmatization)
#
# 3. **Search Index**:
#    - **Fields**: Structured data like title, content, page number
#    - **Analyzers**: Language-specific processing for better text matching
#    - **Scoring Profiles**: Customizing relevance based on specific fields or freshness
#
# 4. **Query Types**:
#    - **Keyword Search**: Direct matching of terms (BM25 algorithm)
#    - **Semantic Search**: Understanding query intent (requires AI models)
#    - **Vector Search**: Finding similar concepts using embeddings
#    - **Filters**: Narrowing results by metadata (e.g., document category)
#
# ### Why Azure AI Search Excels for Knowledge Bases
#
# - **Scale**: Handles millions of documents efficiently
# - **Relevance**: Sophisticated ranking algorithms ensure most relevant content appears first
# - **AI Integration**: Built-in natural language processing capabilities
# - **Security**: Role-based access control and document-level security
# - **Real-time**: Index updates appear in search results immediately
#
# In our agentic RAG system, Azure AI Search forms the foundation of the knowledge retrieval process, allowing the agent to quickly find and leverage the most relevant information from the employee handbook.

# %% [markdown]
# ## 3. Document Indexing with Azure AI Search
#
# Let's set up our document indexing pipeline using Azure AI Search.

# %%
# Define constants for index
INDEX_NAME = "hr-documents"
MAX_TOKENS_PER_CHUNK = 1000
MAX_CHUNKS_PER_DOC = 10
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
# ## 4. Processing the Employee Handbook PDF with Embeddings
#
# Now, let's process the employee handbook PDF file located in the docs folder, generate embeddings, and prepare it for indexing.

# %%
# Function to generate embeddings using Azure OpenAI
async def generate_embeddings(text):
    """Generate embeddings for a text using Azure OpenAI."""
    try:
        # Make direct API call to Azure OpenAI
        response = await embedding_client.embeddings.create(
            input=text,
            model=azure_embedding_deployment
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        # Return a zero vector if there's an error
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
    """Split page content into smaller chunks for better indexing and retrieval and add embeddings."""
    
    chunks = []
    
    for page in pages:
        page_text = page["content"]
        page_num = page["page_num"]
        
        # If the page text is shorter than max_chunk_size, keep it as is
        if len(page_text) <= max_chunk_size:
            # Generate embedding for the chunk
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
                        # Generate embedding for the chunk
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
                # Generate embedding for the chunk
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

# Index documents
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

# %% Process the Employee Handbook PDF
async def process_and_index_pdf():
    pdf_path = "docs/employee_handbook.pdf"
    pdf_pages = extract_text_from_pdf(pdf_path)
    pdf_chunks = await chunk_text_with_embeddings(pdf_pages)
    
    # Index the processed chunks
    indexed_documents = index_documents(pdf_chunks, search_client)
    return indexed_documents

# Setup function for initializing vector search
async def init_vector_search():
    # Process and index the PDF
    await process_and_index_pdf()
    print("Vector search initialized!")
    return True

# When running in Jupyter, you can initialize with:
# await init_vector_search()

# %%
# Define a function to search HR documents using vector search
async def search_hr_documents(query, top=5):
    """
    Search for HR documents based on a query using vector search.
    
    Args:
        query (str): The search query
        top (int): Number of top results to return
        
    Returns:
        str: Formatted search results
    """
    try:
        # Generate embedding for the query
        query_embedding = await generate_embeddings(query)
        
        # Perform vector search
        vector_results = search_client.search(
            search_text=query,
            vector_queries=[{
                "kind": "vector",
                "vector": query_embedding,
                "k": top,
                "fields": "vector"
            }],
            select=["title", "content", "page_num"],
            top=top
        )
        # Format the results
        results_text = f"Search results for: '{query}'\n\n"
        
        for i, result in enumerate(vector_results):
            results_text += f"Result {i+1} (Page {result['page_num']}):\n"
            results_text += f"Title: {result['title']}\n"
            results_text += f"Content: {result['content'][:200]}...\n\n"
        
        return results_text
    
    except Exception as e:
        return f"Error performing search: {str(e)}"

# Let's test the vector search
# await search_hr_documents("What are the company values?", top=3)


# %% [markdown]
# ## 5. Creating Plugins for the Agentic RAG System
#
# Now, let's define a simple search plugin that our agent will use for retrieving information.

# %%
class DocumentSearchPlugin:
    """A Plugin that provides search capabilities for HR documents."""

    def __init__(self, search_client):
        self.search_client = search_client

    @kernel_function(description="Search for HR documents based on a query.")
    async def search_hr_documents(
        self, 
        query: str,
        top: Optional[int] = 3
    ) -> Annotated[str, "Returns the search results as formatted text."]:
        """Search for HR documents that match the query."""
        try:
            # Generate embedding for the query
            query_embedding = await generate_embeddings(query)
            
            # Perform vector search
            vector_results = self.search_client.search(
                search_text=query,
                vector_queries=[{
                    "kind": "vector",
                    "vector": query_embedding,
                    "k": top,
                    "fields": "vector"
                }],
                select=["title", "content", "page_num"],
                top=top
            )
            
            # Format the results
            results_text = f"Search results for: '{query}'\n\n"
            
            for i, result in enumerate(vector_results):
                results_text += f"Result {i+1} (Page {result['page_num']}):\n"
                results_text += f"Title: {result['title']}\n"
                results_text += f"Content: {result['content'][:300]}...\n\n"
            
            return results_text
        
        except Exception as e:
            return f"Error performing search: {str(e)}"

# Register the search plugin with the kernel
kernel.add_plugin(DocumentSearchPlugin(search_client), plugin_name="searchPlugin")

# %% [markdown]
# ## 6. Creating a Simple Agentic RAG Assistant
#
# Let's create a simple agent that can perform multiple searches as needed.

# %%
# Set up the agent with system message that encourages multiple searches
agent = ChatCompletionAgent(
    kernel=kernel,
    instructions="""You are a helpful HR assistant named HRBot that specializes in company policies and procedures.
    Your purpose is to answer employee questions accurately using company documentation.
    
    IMPORTANT SEARCH INSTRUCTIONS:
    1. When answering questions, you have access to a searchPlugin.search_hr_documents function.
    2. You should make MULTIPLE searches with DIFFERENT search queries to gather comprehensive information.
    3. For each question, formulate 2-3 DIFFERENT search queries that approach the question from different angles.
    4. Refine your search queries based on initial results - if information is missing, search again with more specific terms.
    5. When formulating search queries, use HR terminology and specific policy-related keywords.
    
    When responding:
    - Combine information from all search results to provide complete answers
    - Always cite which page of the handbook information comes from
    - If information is unavailable after multiple searches, acknowledge this and suggest who to contact
    - Be professional, concise, and helpful
    
    Example search strategy for "What is the vacation policy?":
    1. First search: "vacation policy allowance"
    2. Second search: "paid time off accrual"
    3. Third search: "requesting vacation procedure"
    """
)

# %% [markdown]
# ## 7. Testing Our Simple Agentic RAG Assistant
#
# Let's test our assistant with some example queries.

# %%
async def test_agentic_search():
    # Create a chat history
    chat_history = ChatHistory()

    user_inputs = [
        "What is the companies mission?",
        "What are the data security policies?",
        "What should I do on my first day at the company?",
    ]

    for user_input in user_inputs:
        # Add the user message to chat history
        chat_history.add_user_message(user_input)
        
        # Display user query
        html_output = f"<p><strong>User:</strong> {user_input}</p>"
        
        agent_name: str | None = None
        full_response = ""
        function_calls = []
        function_results = {}
        
        # Track function calls by their ID and accumulate arguments
        function_call_accumulator = {}

        # Collect the agent's response
        async for content in agent.invoke_stream(chat_history):
            if not agent_name and hasattr(content, 'name'):
                agent_name = content.name

            # Track function calls and results
            for item in content.items:
                if isinstance(item, FunctionCallContent):
                    # Get or create accumulator for this function call
                    call_id = getattr(item, 'id', None) or str(uuid.uuid4())
                    
                    if call_id not in function_call_accumulator:
                        function_call_accumulator[call_id] = {
                            'function_name': item.function_name,
                            'arguments': '',
                            'processed': False
                        }
                    
                    # Accumulate arguments
                    function_call_accumulator[call_id]['arguments'] += item.arguments
                    
                    # Try to parse complete JSON
                    try:
                        args = json.loads(function_call_accumulator[call_id]['arguments'])
                        if not function_call_accumulator[call_id]['processed']:
                            query = args.get("query", "")
                            call_info = f"Calling: search_hr_documents(query=\"{query}\")"
                            function_calls.append(call_info)
                            function_call_accumulator[call_id]['processed'] = True
                    except json.JSONDecodeError:
                        # JSON not complete yet, continue accumulating
                        pass
                        
                elif isinstance(item, FunctionResultContent):
                    result_info = f"Result: {item.result[:150]}..." if len(item.result) > 150 else f"Result: {item.result}"
                    function_calls.append(result_info)
                    # Store function results to add to chat history
                    function_results[item.function_name] = item.result

            # Extract the text content
            if hasattr(content, 'content') and content.content and content.content.strip():
                # Check if this is a regular text message (not function related)
                if not any(isinstance(item, (FunctionCallContent, FunctionResultContent))
                         for item in content.items):
                    full_response += content.content

        # Add function calls to HTML
        if function_calls:
            html_output += '<details><summary style="cursor: pointer; font-weight: bold;">Search Queries (click to expand)</summary><pre>'
            html_output += "\n".join(function_calls)
            html_output += '</pre></details>'

        # Add agent response to HTML
        html_output += f"<p><strong>{agent_name or 'HRBot'}:</strong> {full_response}</p>"
        html_output += "<hr>"

        # Add agent's response to chat history
        if full_response:
            chat_history.add_assistant_message(full_response)

        # Display formatted HTML
        display(Markdown(html_output))

# %%
# await test_agentic_search()

# %% [markdown]
# ## 8. Conclusion
#
# In this challenge, we've explored how to build an agentic RAG system using Azure AI Search and Semantic Kernel. We've learned:
#
# 1. How to extract and process text from PDF documents
# 2. How to index documents in Azure AI Search
# 3. How to create plugins for document retrieval and query refinement
# 4. How to integrate retrieval mechanisms with a conversational agent
# 5. How to improve search results through query refinement
#
# This agentic approach transforms RAG from a simple lookup mechanism into an intelligent system that can handle nuanced information needs by:
#
# - **Dynamically refining queries** to improve search results
# - **Selectively retrieving information** based on the user's needs
# - **Integrating retrieval with other capabilities** through the plugin system
#
# These concepts can be applied to various enterprise scenarios, particularly for building knowledge bases that help employees navigate company policies and procedures. 
