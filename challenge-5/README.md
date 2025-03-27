# Challenge 5: Azure AI Agents SDK - Tools & Agents Simplified

Welcome to Challenge 5 of the AI Agents Hackathon! In this challenge, you'll explore the Azure AI Agents SDK through a series of focused examples that demonstrate its capabilities for building intelligent agents. You'll create practical solutions for employee onboarding and IT support scenarios, showcasing how AI agents can streamline business processes and improve employee experiences.

## Learning Objectives

- Understand the core components and workflow of the Azure AI Agents SDK
- Create conversational agents with varying levels of capability
- Implement Retrieval-Augmented Generation (RAG) using file search tools
- Generate structured data outputs using JSON schema
- Use code interpreter tools for data analysis and visualization
- Build agents that can process documents and extract information
- Apply AI agents to solve real-world business scenarios

## What is Azure AI Agents SDK?

The Azure AI Agents SDK is a client library that allows you to build and run AI agents in Azure. It provides capabilities to:

- Create agents that can execute tasks and answer questions
- Add tools like file search, code interpreter, and custom functions
- Set up conversational threads to maintain context
- Process runs to get agent responses
- Track detailed run steps to understand agent behavior

The SDK follows a consistent workflow pattern:
1. Create an agent with specific instructions and tools
2. Create a thread for conversation context
3. Add messages to the thread
4. Process a run to generate agent responses
5. Retrieve and analyze responses

## Azure AI Agent Service Overview

Azure AI Agent Service is a fully managed service designed to empower developers to securely build, deploy, and scale high-quality, and extensible AI agents without needing to manage the underlying compute and storage resources. What originally took hundreds of lines of code to support client-side function calling can now be done in just a few lines of code with Azure AI Agent Service.

Within Azure AI Foundry, an AI Agent acts as a "smart" microservice that can be used to:

- Answer questions (RAG)
- Perform actions
- Completely automate workflows

It achieves this by combining the power of generative AI models with tools that allow it to access and interact with real-world data sources.

### Benefits Over Direct Model APIs

When compared to developing with the Model Inference API directly, Azure AI Agent Service provides:

- **Automatic tool calling** – No need to parse a tool call, invoke the tool, and handle the response; all of this is now done server-side
- **Securely managed data** – Instead of managing your own conversation state, you can rely on threads to store all the information you need
- **Out-of-the-box tools** – In addition to file retrieval and code interpreter tools, Azure AI Agent Service also provides tools to interact with data sources like Azure AI Search and Azure Functions

### Comparing with Azure OpenAI Assistants

Azure AI Agent Service provides all the capabilities of Azure OpenAI Assistants plus:

- **Flexible model selection** - Create agents using Azure OpenAI models, or others such as Llama 3, Mistral, and Cohere
- **Extensive data integrations** - Ground your AI agents with relevant, secure enterprise knowledge from various data sources
- **Enterprise grade security** - Ensure data privacy and compliance with secure data handling and keyless authentication
- **Storage options** - Either bring your own Azure Blob storage for full control or use platform-managed storage

## What You'll Build

### Getting Started

1. Open the provided Jupyter notebook [5-azure_ai_agents_sdk.ipynb](5-azure_ai_agents_sdk.ipynb)
2. Follow the step-by-step instructions to build your AI agents
3. Complete the exercises and experiments marked throughout the notebook

The notebook guides you through building multiple agents:

1. **Basic Agent for Employee Onboarding**:
   - Create a simple question-answering agent
   - Set up conversation threads and process runs
   - Handle multi-turn conversations with context
   - Learn the fundamental workflow of agent interactions

2. **Agent with File Search (RAG)**:
   - Upload and index company documentation
   - Create vector stores for efficient document retrieval
   - Build an agent that can answer questions based on specific documents
   - Understand the file search process through run steps
   - See how RAG improves answer accuracy with company-specific information

3. **Expense Document Classification Agent with JSON Schema Output**:
   - Extract text from PDF documents using PyPDF2
   - Create agents that return structured data in JSON format
   - Process and classify expense documents
   - Store processed data for further analysis
   - Understand how to use JSON schema for consistent data extraction

4. **Expense Analysis with Code Interpreter**:
   - Analyze processed expense data using the code interpreter tool
   - Generate visualizations and insights dynamically
   - Extract meaningful patterns from financial data
   - See how code interpreter enables advanced data analysis capabilities

## Real-World Applications

The agents you'll build in this challenge have numerous practical applications:

- **Employee Onboarding**: Helping new employees get up to speed quickly with company policies and procedures
- **IT Support**: Providing automated answers to common technical questions and issues
- **Financial Document Processing**: Automating expense report handling and analysis
- **Business Intelligence**: Extracting insights from financial data with minimal manual intervention
- **Document Management**: Organizing and making information accessible through natural language queries

## Resources

- [Azure AI Agent Service Documentation](https://learn.microsoft.com/en-us/azure/ai-services/agents/)
- [What is Azure AI Agent Service?](https://learn.microsoft.com/en-us/azure/ai-services/agents/overview)
- [Azure AI Agent Service Quickstart](https://learn.microsoft.com/en-us/azure/ai-services/agents/quickstart)

## Next Steps

After completing this challenge, you'll have a solid understanding of how to implement AI agents using the Azure AI Agents SDK. You'll understand the core workflow of creating agents, managing conversation threads, and leveraging various tools to enhance agent capabilities. 

This knowledge will be essential for the upcoming challenges where you'll build more sophisticated agents with multiple tools and multi-agent collaboration. You'll be able to apply the patterns you've learned here to create agents tailored to specific business needs and domains.

Ready to move on? [Proceed to Challenge 6](../challenge-6/README.md) to learn about Building an AI Agent with Multiple Tools! 