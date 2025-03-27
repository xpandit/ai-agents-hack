# Challenge 4: Functions & Plugins with Semantic Kernel

Welcome to Challenge 4 of the AI Agents Hackathon! In this challenge, you'll explore Semantic Kernel fundamentals and learn how to create powerful AI agents by combining prompts with custom functions.

## Learning Objectives

- Understand Semantic Kernel core concepts and architecture
- Create reusable functions that cleanly separate logic from prompting
- Develop skills to call custom functions directly from prompts
- Build prompt chains that seamlessly integrate code and AI reasoning
- Implement a practical onboarding FAQ bot using company data

## What is Semantic Kernel?

Semantic Kernel (SK) is an open-source SDK that integrates Large Language Models (LLMs) with programming languages. It provides a structured way to combine AI capabilities with custom code, external data sources, and API services.

What makes Semantic Kernel powerful for building AI agents:

- **Plugin Architecture**: Create reusable components that extend AI capabilities
- **Function Calling**: Allow LLMs to invoke your custom code when needed
- **Seamless Integration**: Combine AI prompts and traditional programming in a unified workflow
- **Memory & Context Management**: Maintain state and manage conversation history
- **Enterprise Readiness**: Designed for production applications with scalability in mind

## Key Concepts in Semantic Kernel

### Plugins

Plugins are collections of related functions that extend what your AI agent can do. There are two types of plugins in Semantic Kernel:

1. **Native Functions**: Traditional code functions written in a programming language (Python, C#, etc.)
2. **Semantic Functions**: Prompt templates that guide the AI to perform specific tasks

This dual approach allows you to leverage both the precision of code and the flexibility of natural language.

### The Power of Function Calling

Function calling creates a two-way bridge between your code and AI:

- **AI → Code**: The AI can call your functions when it determines they're needed
- **Code → AI**: Your code can invoke AI reasoning at specific points in your application flow

This bidirectional relationship creates AI agents that combine the best of both worlds — the reliability and specificity of traditional programming with the adaptability and intelligence of LLMs.

## Hands-on Exercise: Building an Onboarding FAQ Bot

In this challenge, you'll build an intelligent onboarding assistant that helps new employees get up to speed by answering questions using internal company documentation.

### What You'll Create

An AI agent that:
- Retrieves relevant information from company documents
- Answers questions about company policies, benefits, and procedures
- Uses custom functions to format responses, validate information, and access specific data sources
- Maintains conversation context for a natural dialogue experience

### Getting Started

1. Open the provided Jupyter notebook [4-semantic-kernel-plugins.ipynb](4-semantic-kernel-plugins.ipynb)
2. Follow the step-by-step instructions to build your onboarding FAQ bot
3. Complete the exercises and experiments marked throughout the notebook

## Key Implementation Steps

The notebook will guide you through:

1. **Setting Up Semantic Kernel**: Initializing the kernel and configuring LLM services
2. **Creating Native Functions**: Building Python functions that your agent can call
   - Document retrieval function to find relevant information
   - Response formatting function to ensure consistent outputs
   - Validation function to check factual accuracy
   
3. **Designing Semantic Functions**: Creating prompt templates that:
   - Extract question intent
   - Generate comprehensive answers
   - Determine when to call native functions
   
4. **Connecting Everything**: Building the full conversational flow
   - Handling user questions
   - Managing conversation context
   - Orchestrating the interplay between prompts and functions
   
5. **Testing and Refinement**: Evaluating and improving your agent's performance

## Challenge Success Criteria

Your onboarding FAQ bot should:
- Correctly interpret a variety of questions about company policies and procedures
- Retrieve and cite relevant information from the company documents
- Use appropriate custom functions when needed
- Maintain context through a multi-turn conversation
- Present information in a clear, helpful format

## Advanced Concepts (Bonus)

If you complete the main exercise, try extending your agent with:
- Memory management to remember user preferences
- Function chaining for complex multi-step tasks
- Fallback mechanisms when information isn't available
- Personality customization through system messages

## Resources

- [Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/overview/)
- [Function Calling Best Practices](https://learn.microsoft.com/en-us/semantic-kernel/prompts/function-calling/)
- [Plugin Development Guide](https://learn.microsoft.com/en-us/semantic-kernel/agents/plugins/)
- [Python SDK Reference](https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel)

## Next Steps

After completing this challenge, you'll have a solid understanding of how to combine code and AI capabilities using Semantic Kernel's plugin architecture. This foundation will be essential for the upcoming challenges where you'll explore tool usage, RAG, and multi-agent systems.

Ready to move on? [Proceed to Challenge 5](../challenge-5/README.md) to learn about Tool Usage & Agentic RAG! 