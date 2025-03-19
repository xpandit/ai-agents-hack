# Challenge 6: Building an Advanced Single-Agent Scenario

Welcome to Challenge 6 of the AI Agents Hackathon! In this challenge, you'll build a sophisticated IT Helpdesk Assistant that can autonomously handle technical support tickets, decompose complex problems, and output structured data for enterprise systems. You'll have the opportunity to implement this using two different approaches: Microsoft AutoGen and Azure AI Agents Service.

## Learning Objectives

- Design agents with well-defined goals that can operate autonomously
- Implement task decomposition using advanced reasoning capabilities
- Create interfaces between AI agents and mock backend services/APIs
- Develop structured output formats suitable for enterprise software integration
- Build practical workflows for real-world business scenarios
- Compare different agent implementation frameworks

## The IT Helpdesk Assistant Challenge

### Scenario

Your company's IT department is overwhelmed with support tickets. They need an AI assistant that can:

1. Automatically classify incoming tickets by issue type
2. Analyze and break down complex technical problems
3. Generate step-by-step resolution plans
4. Document solutions in a format compatible with the existing ticketing system

### What You'll Create

An advanced IT Helpdesk Assistant that:

- Accepts natural language ticket descriptions from users
- Accurately classifies issues as hardware, software, or network-related
- Decomposes complex problems into manageable subtasks
- Executes a reasoning workflow to provide solutions
- Outputs structured data ready for integration with ticketing systems
- Handles a variety of common IT support scenarios

## Implementation Approaches

In this challenge, you have the flexibility to choose between two powerful frameworks for building your IT Helpdesk Assistant:

### 1. Microsoft AutoGen

AutoGen is an open-source framework for building conversational agents that combines the reasoning capabilities of Large Language Models with practical business logic and operational workflows. Key advantages include:

- **Open-source flexibility**: Full access to customize agent behavior and logic
- **Multi-agent capabilities**: Create specialized agents that can collaborate
- **Local deployment options**: Run entirely within your own infrastructure
- **Extensive customization**: Fine-tune every aspect of agent behavior

### 2. Azure AI Agents Service

Azure AI Agents Service is a fully managed cloud service that simplifies the development and deployment of AI agents. Key advantages include:

- **Streamlined development**: What takes hundreds of lines of code with direct API calls can be done in just a few lines
- **Automatic tool calling**: Server-side handling of function calls, removing the need to manually parse calls and responses
- **Enterprise integration**: Secure connections to Azure services like Azure AI Search and Azure Functions
- **Managed infrastructure**: No need to handle scaling, security, or backend management
- **Multiple model support**: Use various models including Azure OpenAI, Llama 3, Mistral, and Cohere

## Getting Started

This challenge provides two separate Jupyter notebooks:

1. Open `autogen_solution.ipynb` if you want to implement the IT Helpdesk Assistant using Microsoft AutoGen
2. Open `azure_agents_solution.ipynb` if you prefer to use Azure AI Agents Service 

You can choose to complete either one or both approaches based on your interests and available time.

Each notebook contains step-by-step instructions to guide you through:
- Setting up the required environment and dependencies
- Building the core components of your assistant
- Testing with sample IT support tickets
- Extending the solution with advanced features

## Key Implementation Steps

Regardless of which approach you choose, your implementation will include these key components:

1. **Setting Up Your Environment**:
   - Configuring the necessary frameworks and services
   - Understanding agent components and architecture
   - Defining agent goals and system messages

2. **Creating the Ticket Classification System**:
   - Designing a classification schema (hardware/software/network)
   - Building a prompt engineering strategy for accurate classification
   - Implementing validation and confidence checks

3. **Developing Task Decomposition Capabilities**:
   - Creating a problem analysis workflow
   - Implementing subtask generation
   - Building dependency tracking for sequential tasks

4. **Integrating with Mock Backend Services**:
   - Setting up function calling for external tools
   - Creating simulated API endpoints for ticketing systems
   - Implementing structured data validation

5. **Building the Resolution Engine**:
   - Designing solution templates for common problems
   - Implementing step-by-step troubleshooting workflows
   - Creating verification mechanisms for solution quality

6. **Output Formatting for Enterprise Systems**:
   - Defining JSON schemas for ticket documentation
   - Implementing consistent formatting patterns
   - Creating metadata for tracking and analytics

## AutoGen-Specific Implementation

If using AutoGen, you'll focus on:
- Setting up agent definitions and personalities
- Implementing custom reasoning chains
- Creating function-calling interfaces
- Building multi-step troubleshooting workflows

## Azure AI Agents Service-Specific Implementation

If using Azure AI Agents Service, you'll focus on:
- Creating an agent in Azure AI Foundry
- Defining custom tools for IT support functions
- Implementing thread management for conversation tracking
- Handling secure integration with backend systems

## Challenge Success Criteria

Your IT Helpdesk Assistant should:

- Accurately classify at least 90% of sample tickets into the correct category
- Successfully decompose complex problems into logical subtasks
- Generate step-by-step resolution plans that are technically accurate
- Produce structured outputs that conform to the specified JSON schema
- Handle a variety of common IT issues across hardware, software, and networking domains
- Demonstrate reasoning capabilities when approaching unfamiliar problems

## Sample Test Cases

The notebooks include these sample ticket scenarios to test your assistant:

1. **Hardware Issue**: "My laptop won't power on, even when plugged in."
2. **Software Issue**: "MS Word keeps crashing whenever I try to save a document with images."
3. **Network Issue**: "I can connect to the WiFi but can't access any websites or company resources."
4. **Complex Multi-Domain Issue**: "After the recent software update, my laptop runs very slowly, the fan is making unusual noises, and I can't connect to the VPN."

## Advanced Concepts (Bonus)

If you complete the main challenge, try extending your assistant with:

- Prioritization mechanisms based on issue severity and impact
- Personalized solutions based on user technical proficiency
- Automatic generation of preventative maintenance recommendations
- Integration with knowledge base lookups for specialized issues
- Implementation of user feedback loops to improve solutions

## Comparing the Approaches

### When to Choose AutoGen
- When you need maximum flexibility and customization
- For research and experimentation with agent architectures
- When you want to run everything locally or in your own infrastructure
- For projects requiring innovative multi-agent designs

### When to Choose Azure AI Agents Service
- When you need rapid development and deployment
- For enterprise scenarios requiring security and compliance features
- When you want to leverage Azure's managed services ecosystem
- For production systems that need reliable scaling and management

## Resources

### AutoGen Resources
- [Microsoft AutoGen Documentation](https://microsoft.github.io/autogen/)
- [AutoGen Function Calling Guide](https://microsoft.github.io/autogen/docs/Use-Cases/agent_functions)
- [Task Decomposition Patterns](https://microsoft.github.io/autogen/docs/Use-Cases/complex_procedures)
- [Structured Output Formats](https://microsoft.github.io/autogen/docs/tutorials/structured_outputs)

### Azure AI Agents Service Resources
- [Azure AI Agent Service Overview](https://learn.microsoft.com/en-us/azure/ai-services/agents/overview)
- [Getting Started with Azure AI Agent Service](https://learn.microsoft.com/en-us/azure/ai-services/agents/)
- [Tool Integration Guide](https://learn.microsoft.com/en-us/azure/ai-services/agents/)
- [Azure AI Search Integration](https://learn.microsoft.com/en-us/azure/ai-services/agents/)

## Next Steps

After completing this challenge, you'll have hands-on experience building a sophisticated single-agent system using either AutoGen or Azure AI Agents Service (or both). This foundation will prepare you for the final challenge, where you'll create a multi-agent collaborative system.

Ready to move on? [Proceed to Challenge 7](../challenge-7/README.md) to learn about Multi-Agent Collaboration! 