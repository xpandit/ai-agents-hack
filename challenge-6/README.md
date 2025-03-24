# Challenge 6: Tool Usage & Agentic RAG

Welcome to Challenge 5 of the AI Agents Hackathon! In this challenge, you'll explore Retrieval-Augmented Generation (RAG) in an agentic context, building an intelligent assistant that can leverage company documentation to answer queries accurately.

## Learning Objectives

- Understand how RAG can be enhanced with agentic capabilities
- Learn to index and query company documentation using Azure Cognitive Search
- Build intelligent retrieval and generation loops within agent workflows
- Deploy an agentic RAG assistant using Azure AI Agents Service
- Experience how AI agents can make decisions about when and how to retrieve information

## What is Agentic RAG?

Retrieval-Augmented Generation (RAG) enhances Large Language Models by giving them access to external knowledge. Traditional RAG systems follow a linear process: retrieve relevant documents → generate a response based on those documents.

Agentic RAG takes this a step further by adding intelligent decision-making to the retrieval process:

- **Dynamic Query Formulation**: The agent reformulates queries to improve search results
- **Selective Retrieval**: The agent decides when to retrieve information and when to rely on its own knowledge
- **Multi-step Reasoning**: The agent can perform multiple retrieval steps for complex questions
- **Result Evaluation**: The agent assesses the quality and relevance of retrieved information
- **Tool Integration**: The agent combines retrieval with other capabilities (calculations, API calls, etc.)

This agentic approach transforms RAG from a simple lookup mechanism into an intelligent system that can handle nuanced information needs.

## Azure AI Services for Agentic RAG

This challenge will leverage two key Azure services:

### Azure Cognitive Search

A fully managed search service that enables you to:
- Index various document types (PDF, Word, HTML, etc.)
- Create searchable knowledge bases from your content
- Perform semantic search with natural language understanding
- Scale to handle large document collections

### Azure AI Agents Service

A managed service for building and deploying AI agents that:
- Connects to various data sources including Azure Cognitive Search
- Provides built-in tools for document retrieval and analysis
- Supports development of agents with multiple capabilities
- Offers enterprise-grade security and compliance features

## Hands-on Exercise: Building an Agentic RAG Assistant

In this challenge, you'll build an onboarding assistant that can answer employee questions by intelligently retrieving information from HR documentation.

### What You'll Create

An agentic RAG assistant that:
- Indexes a collection of HR documents (policies, benefits, procedures)
- Intelligently retrieves relevant information based on employee questions
- Generates accurate, helpful responses grounded in company documentation
- Makes smart decisions about when to search for information vs. using its baseline knowledge
- Provides a natural conversational experience for new employees

### Getting Started

1. Open the provided Jupyter notebook (`solution.ipynb`)
2. Follow the step-by-step instructions to build your agentic RAG assistant
3. Complete the exercises and experiments marked throughout the notebook

## Key Implementation Steps

The notebook will guide you through:

1. **Setting Up Your Environment**:
   - Configuring Azure services (Cognitive Search, OpenAI)
   - Preparing sample HR documentation
   - Initializing the Azure AI Agents Service client

2. **Building the Document Indexing Pipeline**:
   - Processing and chunking documents
   - Creating and configuring a search index
   - Uploading document chunks to the index
   - Testing basic retrieval functionality

3. **Creating the Agentic RAG Assistant**:
   - Designing the agent's reasoning flow
   - Implementing query refinement strategies
   - Building the retrieval-generation loop
   - Adding citation and source tracking

4. **Testing and Optimization**:
   - Evaluating the assistant with sample questions
   - Analyzing retrieval effectiveness
   - Improving response quality through prompt refinement
   - Measuring performance improvements

## Challenge Success Criteria

Your agentic RAG assistant should:
- Successfully index and retrieve information from HR documents
- Generate accurate responses that reference appropriate sources
- Demonstrate intelligent query reformulation when appropriate
- Show agentic behavior in deciding when to retrieve vs. use baseline knowledge
- Handle a variety of question types related to company policies and procedures
- Maintain context across multiple turns of conversation

## Advanced Concepts (Bonus)

If you complete the main exercise, try extending your agent with:
- Hybrid search combining keyword and semantic approaches
- Multi-document reasoning for complex questions
- Document-specific retrieval strategies based on content type
- Self-evaluation of response quality with automatic correction
- Personalized responses based on employee role or department

## Resources

- [Azure Cognitive Search Documentation](https://learn.microsoft.com/en-us/azure/search/)
- [Azure AI Agents Service](https://learn.microsoft.com/en-us/azure/ai-agents-service/)
- [RAG Pattern Best Practices](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/retrieval-augmented-generation)
- [Semantic Kernel Retrieval Skills](https://learn.microsoft.com/en-us/semantic-kernel/memories/memories-rag)

## Next Steps

After completing this challenge, you'll have a solid understanding of how to implement agentic RAG patterns using Azure services. This knowledge will be essential for the upcoming challenges where you'll build more sophisticated agents with multiple tools and multi-agent collaboration.

Ready to move on? [Proceed to Challenge 7](../challenge-6/README.md) to learn about Multi-Agent Collaboration with Semantic Kernel & AutoGen