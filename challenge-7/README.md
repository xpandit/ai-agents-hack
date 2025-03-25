# Challenge 7: Multi-Agent Collaboration with Semantic Kernel & AutoGen

Welcome to the last challenge of the AI Agents Hackathon! In this culminating challenge, you'll combine all the skills and concepts you've learned to build a sophisticated multi-agent system that leverages the power of both Semantic Kernel and AutoGen frameworks.

## Learning Objectives

- Master multi-agent architecture patterns and understand framework comparisons
- Implement effective communication protocols between collaborative specialized agents
- Create complex integrations involving multiple capabilities (RAG, function calling, task decomposition)
- Design agent orchestration systems that coordinate work between specialized agents
- Build a real-world, enterprise-ready AI solution that solves a practical business problem

## Multi-Agent Systems: Beyond Single-Agent Limitations

Multi-agent systems represent the next evolution in AI application design, addressing limitations that single agents face when handling complex, multi-domain tasks. By creating a team of specialized AI agents that can communicate, collaborate, and coordinate their activities, multi-agent systems can:

- Handle complex problems by breaking them down into specialized domains
- Provide checks and balances through agent debates and verification
- Offer more transparent reasoning through explicit agent interactions
- Scale to handle broader knowledge and skill domains than any single agent could manage
- Mimic human team collaboration for more natural problem-solving approaches

This approach becomes particularly valuable in enterprise settings where tasks often span multiple departments, knowledge bases, and technical domains—exactly the scenario we'll tackle in this challenge.

## Multi-Agent Design Patterns

When implementing multi-agent systems, several design patterns have emerged as effective approaches for different scenarios. Understanding these patterns will help you choose the right architecture for your Employee Onboarding Assistant:

### Group Chat Pattern

In this pattern, multiple agents participate in a shared conversation space, similar to how humans collaborate in messaging platforms. Each agent can:
- See all messages from other agents and users
- Contribute to the conversation based on their expertise
- Build upon information shared by other agents

This pattern works well when semantic information needs to be preserved across multiple conversation turns and when agents need full context to make decisions.

### Hand-Off Pattern

In this pattern, agents sequentially process a request by passing control to the next specialized agent in a chain:
- Each agent handles a specific aspect of the request
- The conversation flows from one agent to another in a predetermined sequence
- Each agent adds its specialized knowledge before passing control

This works well for workflows with clear, sequential steps that require different domain expertise at each stage.

### Collaborative Filtering Pattern

In this pattern, multiple specialized agents analyze the same request independently and their outputs are combined or filtered:
- Each agent applies its unique expertise and perspective
- A coordinator agent synthesizes the various outputs
- Conflicting recommendations can be resolved through voting or priority rules

This pattern is particularly useful when diverse expertise is needed to make comprehensive recommendations.

### Visibility and Monitoring

Any multi-agent system requires mechanisms to monitor and debug agent interactions:
- **Logging**: Track agent actions, decisions, and handoffs
- **Visualization**: Represent agent interactions graphically to identify bottlenecks
- **Performance Metrics**: Measure response times, accuracy, and user satisfaction

By implementing proper monitoring, you can ensure your multi-agent system is functioning effectively and identify areas for improvement.

## The Convergence of Frameworks

This challenge explores the exciting convergence of Microsoft's agentic AI frameworks. As announced by Microsoft, there are plans to align the multi-agent runtime in AutoGen with Semantic Kernel in early 2025, creating a unified approach for AI agent development.

While we await this official convergence, we can already leverage the strengths of both frameworks in complementary ways:

- **Semantic Kernel**: A production-ready SDK (v1.0+ across .NET, Python, and Java) designed for enterprise AI applications. It excels at:
  - Enterprise integration with robust security and stability
  - Plugin architecture for extending functionality
  - Structured agent development with the Agent Framework
  - Process orchestration for business workflows
  - Multi-agent coordination through group chat capabilities
  - Strong typing and enterprise-grade deployment patterns

- **AutoGen**: A research-oriented framework maintained by Microsoft Research's AI Frontiers Lab, specializing in:
  - Advanced conversable agent design patterns
  - Flexible agent-to-agent communication protocols
  - Dynamic group chat orchestration with various topologies
  - Code execution and tool use within agent conversations
  - Cutting-edge research on agentic capabilities
  - Highly customizable agent behaviors and interactions

By understanding and leveraging both frameworks, you'll be well-positioned to create sophisticated multi-agent systems today while preparing for their future convergence.

## Employee Onboarding Assistant System

### Scenario

Imagine your company is growing rapidly and HR is struggling to handle the influx of new employee questions during onboarding. To solve this, you'll create a comprehensive Employee Onboarding Assistant System that consists of multiple specialized agents working together to provide a seamless experience for new hires.

### System Architecture

An example multi-agent system will feature:

1. **Coordinator Agent**: The central orchestrator that interacts directly with users, routes queries to specialized agents, and synthesizes final responses
   
2. **Specialized Sub-Agents**:
   - **IT Support Agent**: Handles technical setup questions and access requests (based on Challenge 6)
   - **HR Policy Agent**: Answers questions about company policies, benefits, and procedures using RAG (based on Challenge 5)
   - **Facilities Agent**: Manages office-related queries like desk assignments, building access, and equipment
   - **Training Agent**: Provides information about required training and development resources

3. **Utility Agents**:
   - **Research Agent**: Retrieves and summarizes information from company knowledge bases when needed
   - **Verification Agent**: Validates the accuracy of responses before they reach the user
   - **Reasoning Agent**: Helps decompose complex requests into manageable sub-tasks

### Communication Flows

Your system will implement sophisticated communication protocols:
- **Query routing**: Determining which agent(s) should handle a specific request
- **Information sharing**: Passing context and knowledge between agents
- **Consensus building**: Resolving conflicting information or recommendations
- **Response synthesis**: Creating cohesive answers from multiple agent inputs

## Implementation Options

This challenge offers three implementation paths, each with its own approach to multi-agent architecture:

### Option 1: Semantic Kernel Multi-Agent System

Use Semantic Kernel's agent framework to build a system that emphasizes enterprise integration and structured workflows:

- **Agent Architecture**: Define agents using Semantic Kernel's `ChatCompletionAgent` or `OpenAIAssistantAgent` classes, each with specific instructions and capabilities
- **Communication Pattern**: Implement the `AgentGroupChat` for orchestrating conversations between multiple specialized agents
- **Tool Integration**: Develop custom plugins that provide domain-specific functionality to each agent
- **State Management**: Utilize Semantic Kernel's context management for maintaining conversation state
- **Execution Flow**: Configure flexible termination and selection strategies to determine when conversations end and which agent responds next

Example agent initialization in Python:
```python
# Import necessary modules
from semantic_kernel import Kernel
from semantic_kernel.chat_completion import ChatHistory
from semantic_kernel.agents import ChatCompletionAgent, OpenAIAssistantAgent, AgentGroupChat
from semantic_kernel.agents.chat import ChatMessageContent, AuthorRole
from semantic_kernel.agents.termination_strategies import ApprovalTerminationStrategy
from semantic_kernel.connectors.ai import AzureChatCompletion

# Initialize kernel
kernel = Kernel()

# Add AI service
kernel.add_service(AzureChatCompletion(service_id="azure_chat"))

# Add custom plugins for domain-specific functionality
kernel.add_plugin(plugin=HRPolicyPlugin(), plugin_name="hr_policy")
kernel.add_plugin(plugin=ITSupportPlugin(), plugin_name="it_support")

# Create specialized agents
hr_agent = ChatCompletionAgent(
    service_id="azure_chat",
    kernel=kernel,
    name="HR Policy Agent",
    instructions="You are an expert on company HR policies. Help new employees understand benefits, policies, and procedures."
)

it_agent = ChatCompletionAgent(
    service_id="azure_chat",
    kernel=kernel,
    name="IT Support Agent",
    instructions="You are an IT support specialist. Help new employees set up their accounts, equipment, and resolve technical issues."
)

coordinator_agent = ChatCompletionAgent(
    service_id="azure_chat",
    kernel=kernel,
    name="Coordinator",
    instructions="You are the coordinator who manages the conversation. Review answers from other agents and approve them when they're complete and accurate."
)

# Set up agent group chat with termination strategy
group_chat = AgentGroupChat(
    agents=[hr_agent, it_agent, coordinator_agent],
    termination_strategy=ApprovalTerminationStrategy(agents=[coordinator_agent])
)

# Add user query to chat
group_chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content="I need to set up my laptop and enroll in benefits. What should I do first?"))

# Process conversation and display messages
async for message in group_chat.invoke():
    print(f"{message.author.name}: {message.content}")
```

### Option 2: AutoGen Multi-Agent System

Use AutoGen's collaborative agent framework to create a more research-oriented system with dynamic conversation patterns:

- **Agent Architecture**: Build with AutoGen's `AssistantAgent` and `UserProxyAgent` classes that support autonomous conversations
- **Communication Pattern**: Configure GroupChat for flexible agent interactions with customizable speaker selection
- **Tool Integration**: Leverage function calling for external tools and code execution
- **State Management**: Implement customizable message handling and state tracking
- **Execution Flow**: Define conversation flow using termination conditions and message routing logic

Example agent setup in Python:
```python
# Import necessary AutoGen modules
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import json

# Define agent configurations
llm_config = {
    "config_list": [{"model": "gpt-4", "api_key": "your-api-key"}],
    "temperature": 0.7,
    "timeout": 600,
}

# Create specialized agents
hr_agent = AssistantAgent(
    name="HR_Policy_Expert",
    system_message="You are an expert on company HR policies. Answer questions about benefits, policies, and procedures for new employees.",
    llm_config=llm_config,
)

it_agent = AssistantAgent(
    name="IT_Support_Specialist",
    system_message="You are an IT specialist. Help new employees with technical setup, account access, and equipment requests.",
    llm_config=llm_config,
)

facilities_agent = AssistantAgent(
    name="Facilities_Manager",
    system_message="You manage office facilities. Answer questions about desk assignments, building access, and office equipment.",
    llm_config=llm_config,
)

# Create a coordinator agent
coordinator = AssistantAgent(
    name="Onboarding_Coordinator",
    system_message="You are the coordinator. Route queries to the appropriate specialized agent, synthesize information, and ensure complete responses.",
    llm_config=llm_config,
)

# Create a user proxy agent that can execute functions
user_proxy = UserProxyAgent(
    name="New_Employee",
    human_input_mode="NEVER",  # For automation, no human input needed
    system_message="You are a new employee asking onboarding questions.",
    code_execution_config={"last_n_messages": 3, "work_dir": "coding"},
)

# Set up a group chat with all agents
groupchat = GroupChat(
    agents=[user_proxy, coordinator, hr_agent, it_agent, facilities_agent],
    messages=[],
    max_round=10,  # Limit conversation rounds
)

# Create a manager to handle the conversation flow
manager = GroupChatManager(
    groupchat=groupchat, 
    llm_config=llm_config,
)

# Initiate the conversation with a query
user_message = "I'm starting next Monday and need to know what to bring on my first day, how to get building access, and when I'll receive my laptop."
chat_result = manager.initiate_chat(
    user_proxy,
    message=user_message
)

# The conversation will automatically flow between the agents
```

### Option 3: Hybrid Approach (Advanced)

For those seeking an extra challenge, explore the integration of both frameworks:

- Use AutoGen for complex agent conversation patterns
- Leverage Semantic Kernel for enterprise integration and structured plugin development
- Create bridge components that allow the frameworks to communicate
- Implement patterns that anticipate the planned framework convergence
- Extend both frameworks with custom components that address specific needs

This approach requires deeper understanding of both frameworks but allows you to combine AutoGen's flexible agent interactions with Semantic Kernel's enterprise-ready features.

Example hybrid implementation approach:

```python
# Import libraries from both frameworks
from semantic_kernel import Kernel as SKKernel
from semantic_kernel.plugins.core import web_search_plugin
from semantic_kernel.connectors.ai import AzureChatCompletion
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Initialize Semantic Kernel
sk_kernel = SKKernel()
sk_kernel.add_service(AzureChatCompletion(service_id="azure_chat"))

# Create Semantic Kernel plugins for specialized functionality
sk_kernel.add_plugin(web_search_plugin, "WebSearch")

# Define a custom bridge to connect frameworks
class SemanticKernelBridge:
    def __init__(self, kernel):
        self.kernel = kernel
        
    async def execute_plugin(self, plugin_name, function_name, input_text):
        plugin = self.kernel.plugins.get(plugin_name)
        function = plugin.functions.get(function_name)
        result = await self.kernel.invoke(function, input_text)
        return str(result)

# Initialize the bridge
sk_bridge = SemanticKernelBridge(sk_kernel)

# Define AutoGen agent configurations with access to Semantic Kernel functionality
llm_config = {
    "config_list": [{"model": "gpt-4", "api_key": "your-api-key"}],
    "functions": [
        {
            "name": "search_company_knowledge_base",
            "description": "Search internal company knowledge base using web search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    ]
}

# Create agent that can use Semantic Kernel plugins via the bridge
research_agent = AssistantAgent(
    name="Research_Specialist",
    system_message="You research company information. Use the search_company_knowledge_base function when needed.",
    llm_config=llm_config,
)

# Create user proxy that can execute Semantic Kernel functions
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config={"last_n_messages": 3, "work_dir": "coding"},
)

# Register custom function to connect frameworks
async def search_company_knowledge_base(query):
    result = await sk_bridge.execute_plugin("WebSearch", "search", query)
    return f"Search results for '{query}': {result}"

user_proxy.register_function(search_company_knowledge_base)

# Set up the conversation flow
groupchat = GroupChat(
    agents=[user_proxy, research_agent],
    messages=[],
    max_round=10,
)

manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Start the conversation
user_message = "What's our company policy on remote work?"
chat_result = manager.initiate_chat(user_proxy, message=user_message)
```

This hybrid approach allows you to:
1. Leverage Semantic Kernel's advanced plugin system and enterprise integrations
2. Use AutoGen's flexible agent communication patterns
3. Share information across frameworks through custom bridge components
4. Benefit from the strengths of both systems while mitigating their individual limitations

## Key Implementation Steps

### 1. Setting Up the Environment
- Configure the frameworks and services
- Define agent roles and communication protocols
- Establish the core conversation architecture

### 2. Building Specialized Agents
- Implement each agent with specific domain knowledge and capabilities
- Create the necessary plugins or tools for each agent
- Design appropriate prompts and system messages

### 3. Developing Communication Mechanisms
- Implement message routing between agents
- Create protocols for sharing context and information
- Design consensus-building mechanisms for conflicting information

### 4. Integrating RAG Capabilities
- Connect agents to company knowledge bases
- Implement retrieval strategies for different document types
- Create relevance assessment mechanisms

### 5. Implementing User Interface
- Design conversation flows from the user perspective
- Create natural transitions between agent specialists
- Implement context preservation across the conversation

### 6. Testing and Refinement
- Evaluate the system with realistic onboarding scenarios
- Measure response quality, accuracy, and helpfulness
- Optimize prompt engineering and agent interactions

## Challenge Success Criteria

Your Employee Onboarding Assistant System should:

- Correctly route user queries to the appropriate specialized agents
- Enable collaborative problem-solving between multiple agents
- Retrieve relevant information from knowledge bases when needed
- Generate helpful, accurate responses for common onboarding questions
- Handle complex queries that span multiple domains
- Present a coherent, seamless experience to the user, despite the complexity behind the scenes

## Sample Test Scenarios

Test your system with these realistic onboarding queries:

1. **Cross-domain query**: "I'm starting next Monday and need to know what to bring on my first day, how to get building access, and when I'll receive my laptop."

2. **Complex IT setup**: "I need help setting up my development environment with the right permissions to access the customer database for my project."

3. **Policy clarification**: "What's the company policy on remote work, and how do I request equipment for my home office?"

4. **Training requirements**: "Which mandatory training courses do I need to complete in my first week, and how do I access them?"

5. **Multi-step process**: "I need to set up direct deposit for my paycheck, choose my health insurance plan, and schedule my orientation. Where do I start?"

## Evaluating Your Multi-Agent System

Developing an effective multi-agent system requires careful evaluation beyond just functional testing. Here are key dimensions to assess your implementation:

### Agent Communication Quality
- **Information Flow**: Is critical information successfully passed between agents?
- **Redundancy**: Do agents repeat each other unnecessarily?
- **Handoffs**: Are transitions between agents smooth and contextually appropriate?
- **Conflict Resolution**: How do agents handle disagreements about facts or approaches?

### System Robustness
- **Error Handling**: Does the system gracefully recover from agent failures?
- **Out-of-Scope Queries**: How does the system respond to questions outside its knowledge domain?
- **Ambiguity Management**: Can the system request clarification when user queries are unclear?
- **Consistency**: Are responses consistent across similar queries?

### User Experience
- **Response Quality**: Are answers accurate, helpful, and appropriately detailed?
- **Response Time**: How quickly does the system produce complete answers?
- **Conversation Flow**: Does the interaction feel natural and coherent?
- **Transparency**: Is it clear to users when different agents are involved?

Consider implementing logging and metrics collection to track these dimensions systematically. For example, you might log agent-to-agent messages, measure response times, or implement a feedback mechanism to rate answer quality.

## Advanced Concepts (Bonus)

If you complete the main challenge, extend your system with:

- **Adaptive learning**: Agents that improve their responses based on user feedback
- **Memory mechanisms**: Long-term context tracking across multiple user sessions
- **Explanation capabilities**: Agents that can explain their reasoning and decision processes
- **Fallback strategies**: Elegant handling of edge cases and out-of-scope requests
- **Multi-modal support**: Incorporating images and document uploads into the conversation flow

## Resources

### Semantic Kernel Multi-Agent Resources
- [Introducing Agents in Semantic Kernel](https://devblogs.microsoft.com/semantic-kernel/introducing-agents-in-semantic-kernel/)
- [Working with Agents in Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/)
- [Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/)

### AutoGen Multi-Agent Resources
- [AutoGen Multi-Agent Systems](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat)
- [AutoGen Documentation](https://microsoft.github.io/autogen/stable/)
- [AutoGen Tutorials](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/)

### Framework Integration Resources
- [The Future of AI: Exploring Multi-Agent AI Systems](https://techcommunity.microsoft.com/blog/aiplatformblog/the-future-of-ai-exploring-multi-agent-ai-systems/4226593)
- [Microsoft's Agentic Frameworks: AutoGen and Semantic Kernel](https://devblogs.microsoft.com/autogen/microsofts-agentic-frameworks-autogen-and-semantic-kernel/)
- [Creating Multi-Agent Systems with Semantic Kernel and Azure AI Agent Service](https://medium.com/data-science-collective/create-multi-agent-system-with-microsofts-azure-ai-agent-service-and-semantic-kernel-framework-in-a6c68b123e54)

## Conclusion

This capstone challenge brings together all the concepts you've learned throughout the hackathon into a sophisticated, real-world AI solution. By building a multi-agent Employee Onboarding Assistant, you'll demonstrate mastery of advanced AI orchestration techniques and create a system that showcases the power of collaborative AI agents.

The multi-agent architecture you'll develop represents the cutting edge of AI application design, pointing toward a future where specialized AI agents work together seamlessly to solve complex problems that no single agent could tackle alone.

Congratulations on reaching this final challenge, and good luck creating your multi-agent system! 