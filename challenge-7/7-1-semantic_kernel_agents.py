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
# # Challenge 7-1: Multi-Agent Collaboration with Semantic Kernel
#
# In this notebook, we'll implement a multi-agent collaboration system using Microsoft's Semantic Kernel framework. We'll create an Employee Onboarding Assistant System where multiple specialized agents work together to provide a seamless experience for new hires.
#
# ## What is Semantic Kernel?
#
# Semantic Kernel is a production-ready SDK (v1.0+ across .NET, Python, and Java) designed for enterprise AI applications. It excels at:
#
# - Enterprise integration with robust security and stability
# - Plugin architecture for extending functionality
# - Structured agent development with the Agent Framework
# - Process orchestration for business workflows
# - Multi-agent coordination through group chat capabilities
# - Strong typing and enterprise-grade deployment patterns
#
# ## Multi-Agent Systems Overview
#
# Multi-agent systems represent the next evolution in AI application design, addressing limitations that single agents face when handling complex, multi-domain tasks. By creating a team of specialized AI agents that can communicate, collaborate, and coordinate their activities, multi-agent systems can:
#
# - Handle complex problems by breaking them down into specialized domains
# - Provide checks and balances through agent debates and verification
# - Offer more transparent reasoning through explicit agent interactions
# - Scale to handle broader knowledge and skill domains than any single agent could manage
# - Mimic human team collaboration for more natural problem-solving approaches
#
# In this notebook, we'll implement a multi-agent system using Semantic Kernel's Agent Framework, specifically leveraging the Group Chat functionality to create a team of specialized agents for employee onboarding.

# %% [markdown]
# ## Setting up the Environment
#
# First, let's install the necessary packages and set up our environment. For this multi-agent system, we'll need:
#
# - **semantic-kernel**: The core framework for building our agents and plugins
# - **python-dotenv**: For managing API keys and environment variables
# - **openai**: To interact with OpenAI models (or Azure OpenAI services)
# - **matplotlib**: For visualization (if needed)
#
# These packages allow us to create a secure, flexible foundation for our multi-agent system.

# %%
# Install required packages
# !pip install semantic-kernel python-dotenv openai matplotlib

# %%
# Import required libraries
import os
import json
import asyncio
import uuid
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Import Semantic Kernel components
import semantic_kernel as sk
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat
from semantic_kernel.agents.strategies import DefaultTerminationStrategy
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from typing import Annotated
from semantic_kernel.functions import kernel_function


# Load environment variables from .env file
load_dotenv()

# %% [markdown]
# ## Configuring the Semantic Kernel
#
# Next, we'll configure the Semantic Kernel with our API settings. The kernel serves as the central orchestrator for our agents and functions. It manages connections to AI models, handles function registration, and coordinates the interactions between components.
#
# For enterprise scenarios, you'll typically use Azure OpenAI services for:
# - Enhanced security and compliance features
# - Data residency controls
# - Service level agreements (SLAs)
# - Integration with other Azure services
#
# In this example, we'll configure the kernel to use Azure OpenAI, but Semantic Kernel also supports direct OpenAI integration.

# %%
# Initialize Semantic Kernel
kernel = sk.Kernel()

service_id = "azure_chat"

# Azure OpenAI configuration
kernel.add_service(
    AzureChatCompletion(
        service_id=service_id,
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    )
)

# %% [markdown]
# ## Creating Custom Plugins for Specialized Domains
#
# In Semantic Kernel, plugins are collections of related functions that extend agent capabilities with domain-specific functionality. They allow us to:
#
# - Organize related functionality in logical groups
# - Encapsulate domain knowledge in reusable components 
# - Define clear interfaces between AI and business logic
# - Enable agents to access real-world data and systems
#
# For our onboarding system, we'll create plugins for four domains:
#
# 1. **HR Policy Plugin**: Contains functions for benefits, policies, and procedures
# 2. **IT Support Plugin**: Handles technical setup, equipment, and system access
# 3. **Facilities Plugin**: Manages office locations, amenities, and desk assignments
# 4. **Training Plugin**: Provides information about training and development resources
#
# Each plugin implements the `kernel_function` decorator to expose its methods to the kernel and make them available to our agents.

# %%
class HRPolicyPlugin:
    """Plugin for handling HR policies and questions."""
    
    @kernel_function(description="Get information about employee benefits.")
    def get_benefits_info(self, benefit_type: Optional[Annotated[str, "Type of benefit to get information about (health, retirement, pto, parental)"]] = None) -> str:
        """Get information about employee benefits."""
        benefits = {
            "health": "Contoso offers comprehensive health insurance with dental and vision coverage.",
            "retirement": "Employees can enroll in the 401(k) plan with 5% company matching.",
            "pto": "New employees start with 15 days of PTO plus 10 holidays annually.",
            "parental": "Contoso provides 12 weeks of paid parental leave for all new parents."
        }
        
        if benefit_type and benefit_type.lower() in benefits:
            return benefits[benefit_type.lower()]
        else:
            return "\n".join([f"- {k.capitalize()}: {v}" for k, v in benefits.items()])
    
    @kernel_function(description="Get the onboarding checklist for new employees.")
    def get_onboarding_checklist(self) -> str:
        """Get the onboarding checklist for new employees."""
        checklist = [
            "Complete I-9 and tax forms",
            "Enroll in benefits plans (within 30 days)",
            "Set up direct deposit for payroll",
            "Complete required compliance training",
            "Schedule orientation session with HR",
            "Review employee handbook"
        ]
        return "\n".join([f"- {item}" for item in checklist])
    
    @kernel_function(description="Get information about company policies.")
    def get_company_policies(self, policy_type: Optional[Annotated[str, "Type of policy to get information about (remote, dress, expenses, travel)"]] = None) -> str:
        """Get information about company policies."""
        policies = {
            "remote": "Contoso has a hybrid work policy allowing 3 days of remote work per week.",
            "dress": "Business casual dress code is standard, with casual Fridays.",
            "expenses": "All business expenses require manager approval and receipts.",
            "travel": "Travel must be booked through the company travel portal."
        }
        
        if policy_type and policy_type.lower() in policies:
            return policies[policy_type.lower()]
        else:
            return "\n".join([f"- {k.capitalize()}: {v}" for k, v in policies.items()])

# %% [markdown]
# Now, let's create the IT Support Plugin. This plugin will handle technical questions about equipment, software, and system access.

# %%
# Create an IT Support Plugin
class ITSupportPlugin:
    """Plugin for handling IT support and technical setup questions."""
    
    @kernel_function(description="Get information about standard equipment provided to employees.")
    def get_equipment_info(self, equipment_type: Optional[Annotated[str, "Type of equipment to get information about (laptop, monitors, accessories, mobile)"]] = None) -> str:
        """Get information about standard equipment provided to employees."""
        equipment = {
            "laptop": "Standard laptops are Dell XPS 13 or MacBook Pro 14\".",
            "monitors": "Each employee receives two 27\" 4K monitors.",
            "accessories": "Standard accessories include keyboard, mouse, headset, and docking station.",
            "mobile": "Employees may choose between iPhone or Android devices."
        }
        
        if equipment_type and equipment_type.lower() in equipment:
            return equipment[equipment_type.lower()]
        else:
            return "\n".join([f"- {k.capitalize()}: {v}" for k, v in equipment.items()])
    
    @kernel_function(description="Get information about standard software and access.")
    def get_software_access(self) -> str:
        """Get information about standard software and access."""
        software = [
            "Microsoft 365 (including email, Teams, OneDrive)",
            "Slack for team communication",
            "Jira for project management",
            "GitHub for code repositories",
            "Zoom for video conferencing"
        ]
        return "\n".join([f"- {item}" for item in software])
    
    @kernel_function(description="Get setup instructions for company systems.")
    def get_setup_instructions(self, system_type: Annotated[str, "Type of system to get setup instructions for (windows, mac)"] = "windows") -> str:
        """Get setup instructions for company systems."""
        if system_type.lower() == "mac":
            return """
            Mac Setup Instructions:
            1. Power on your Mac and follow initial setup
            2. Connect to the company WiFi: "Contoso-Secure" (password will be provided)
            3. Install company certificate from IT portal
            4. Set up VPN using GlobalProtect (instructions in IT portal)
            5. Login to company portal with your credentials to install required software
            """
        else:
            return """
            Windows Setup Instructions:
            1. Power on your Windows laptop and login with provided credentials
            2. Connect to the company WiFi: "Contoso-Secure" (password will be provided)
            3. Launch the IT Setup application from the desktop
            4. Follow the prompts to install required software and security tools
            5. Set up VPN using GlobalProtect (instructions in IT portal)
            """

# %% [markdown]
# The Facilities Plugin will handle questions about physical workspaces, office locations, and amenities. This is especially important for helping new employees navigate their physical work environment.

# %%
# Create a Facilities Plugin
class FacilitiesPlugin:
    """Plugin for handling facilities and office-related questions."""
    
    @kernel_function(description="Get information about office locations.")
    def get_office_locations(self) -> str:
        """Get information about office locations."""
        locations = {
            "Seattle": "Main HQ: 123 Tech Blvd, Seattle, WA",
            "San Francisco": "West Coast Office: 456 Innovation St, San Francisco, CA",
            "New York": "East Coast Office: 789 Enterprise Ave, New York, NY",
            "London": "European HQ: 101 Digital Ln, London, UK"
        }
        return "\n".join([f"- {k}: {v}" for k, v in locations.items()])
    
    @kernel_function(description="Get information about office amenities.")
    def get_office_amenities(self, location: Annotated[Optional[str], "Specific office location to get amenities for"] = None) -> str:
        """Get information about office amenities."""
        amenities = {
            "Seattle": ["On-site gym", "Cafeteria", "Game room", "Rooftop lounge", "Bike storage"],
            "San Francisco": ["Coffee bar", "Meditation room", "Gym discount", "Rooftop garden"],
            "New York": ["On-site cafe", "Fitness center", "Library", "Meditation room"],
            "London": ["Tea room", "Fitness center", "Cafeteria", "Game area"]
        }
        
        if location and location.title() in amenities:
            return "\n".join([f"- {item}" for item in amenities[location.title()]])
        else:
            return "All Contoso offices feature modern workspaces, break areas, meeting rooms, and video conferencing facilities."
    
    @kernel_function(description="Get information about desk assignment process.")
    def get_desk_assignment_process(self) -> str:
        """Get information about desk assignment process."""
        return """
        Desk Assignment Process:
        1. Your team lead will show you to your assigned desk on your first day
        2. Your desk will be labeled with your name
        3. Standard equipment will be set up and waiting for you
        4. If you need any special accommodations, please notify HR in advance
        5. Hot-desking is available in designated areas for flexible work
        """

# %% [markdown]
# Finally, let's create the Training Plugin to manage learning and development resources. This plugin helps new employees understand required training and access educational resources.

# %%
# Create a Training Plugin
class TrainingPlugin:
    """Plugin for handling training and development questions."""
    
    @kernel_function(description="Get information about required training.")
    def get_required_training(self, department: Annotated[Optional[str], "The department to get specific training for"] = None) -> str:
        """Get information about required training."""
        general_training = [
            "Security and Compliance (all employees)",
            "Anti-Harassment Policy (all employees)",
            "Company Values and Culture (all employees)"
        ]
        
        department_specific = {
            "engineering": ["Secure Coding Practices", "Code Review Process", "DevOps Pipeline Training"],
            "sales": ["CRM System Training", "Sales Methodology", "Product Knowledge Certification"],
            "hr": ["HRIS System Training", "Employment Law Basics", "Benefits Administration"],
            "marketing": ["Marketing Platform Training", "Brand Guidelines", "Analytics Tools"]
        }
        
        if department and department.lower() in department_specific:
            result = "Required training:\n"
            result += "\n".join([f"- {item}" for item in general_training])
            result += "\n\nDepartment-specific training:\n"
            result += "\n".join([f"- {item}" for item in department_specific[department.lower()]])
            return result
        else:
            return "Required training for all employees:\n" + "\n".join([f"- {item}" for item in general_training])
    
    @kernel_function(description="Get information about available learning resources.")
    def get_learning_resources(self) -> str:
        """Get information about learning resources."""
        resources = [
            "Contoso Learning Portal: https://learn.contoso.com",
            "LinkedIn Learning (free access for employees)",
            "Department-specific documentation libraries",
            "Weekly lunch-and-learn sessions",
            "Quarterly development workshops"
        ]
        return "\n".join([f"- {item}" for item in resources])
    
    @kernel_function(description="Get information about the company mentor program.")
    def get_mentor_program_info(self) -> str:
        """Get information about the mentor program."""
        return """
        Contoso Mentor Program:
        - Each new employee is paired with an experienced mentor
        - Mentors provide guidance, answer questions, and help with networking
        - Mentor relationships typically last 3-6 months
        - Mentor assignments are made during your first week
        - Monthly check-ins are encouraged to track progress
        """

# %% [markdown]
# ## Creating Specialized Agents
#
# Now that we have our plugins defined, we'll create specialized agents that leverage these plugins. Each agent will have:
#
# 1. **A specific domain of expertise**: A focused area of knowledge and responsibility
# 2. **Access to relevant plugins**: Giving them the ability to retrieve accurate information
# 3. **Custom instructions**: Guiding their behavior, tone, and interaction style
# 4. **A clear role definition**: Establishing their purpose in the multi-agent system
#
# Semantic Kernel's `ChatCompletionAgent` provides a powerful foundation for building these specialized AI assistants. By giving each agent targeted knowledge and capabilities, we create a system that can handle complex, multi-domain tasks more effectively than a single general-purpose agent.
#
# First, let's create our HR Policy Agent - responsible for benefits, policies, and procedures:

# %%
# Create HR Policy Agent
hr_agent = ChatCompletionAgent(
    kernel=kernel,
    name="hr-policy-agent",
    description="Expert on company HR policies, benefits, and procedures",
    instructions="""
    You are an HR Policy Expert for Contoso Electronics.
    
    Your role is to help new employees understand:
    - Company benefits and how to enroll
    - HR policies and procedures
    - Required documentation and forms
    - Onboarding processes and checklists
    
    Use the hr_policy plugin functions to provide accurate information.
    If asked about topics outside your expertise, defer to the appropriate specialized agent.
    
    Always be helpful, informative, and welcoming to new employees.
    """,
    plugins=[HRPolicyPlugin()]
)

# %% [markdown]
# Next, let's create the IT Support Agent. This agent will handle technical questions about equipment, software, and system access:

# %%
# Create IT Support Agent
it_agent = ChatCompletionAgent(
    kernel=kernel,
    name="it-support-agent",
    description="Expert on technical setup, equipment, and system access",
    instructions="""
    You are an IT Support Specialist for Contoso Electronics.
    
    Your role is to help new employees with:
    - Setting up their computers and equipment
    - Accessing company systems and software
    - Understanding IT policies and security requirements
    - Troubleshooting common technical issues
    
    Use the it_support plugin functions to provide accurate information.
    If asked about topics outside your expertise, defer to the appropriate specialized agent.
    
    Provide clear, step-by-step instructions and be patient with technical questions.
    """,
    plugins=[ITSupportPlugin()]
)

# %% [markdown]
# Now, let's create the Facilities Agent to handle questions about offices, workspaces, and amenities:

# %%
# Create Facilities Agent
facilities_agent = ChatCompletionAgent(
    kernel=kernel,
    name="facilities-agent",
    description="Expert on office locations, amenities, and desk assignments",
    instructions="""
    You are a Facilities Coordinator for Contoso Electronics.
    
    Your role is to help new employees with:
    - Office locations and layouts
    - Building access and security
    - Desk assignments and workspaces
    - Office amenities and resources
    - Parking and transportation options
    
    Use the facilities plugin functions to provide accurate information.
    If asked about topics outside your expertise, defer to the appropriate specialized agent.
    
    Focus on making new employees feel comfortable in their physical work environment.
    """,
    plugins=[FacilitiesPlugin()]
)

# %% [markdown]
# Finally, let's create the Training Agent to handle learning and development questions:

# %%
# Create Training Agent
training_agent = ChatCompletionAgent(
    kernel=kernel,
    name="training-agent",
    description="Expert on required training and learning resources",
    instructions="""
    You are a Training and Development Specialist for Contoso Electronics.
    
    Your role is to help new employees with:
    - Required onboarding training modules
    - Learning resources and platforms
    - Professional development opportunities
    - Mentorship programs
    - Department-specific training
    
    Use the training plugin functions to provide accurate information.
    If asked about topics outside your expertise, defer to the appropriate specialized agent.
    
    Be encouraging and supportive of employees' learning journeys.
    """,
    plugins=[TrainingPlugin()]
)

# %% [markdown]
# In addition to the specialized agents, we need a coordinator to orchestrate the conversation and ensure a coherent user experience. The coordinator agent serves as:
#
# 1. A traffic director - routing questions to the appropriate specialized agents
# 2. A synthesizer - combining information from multiple sources into coherent responses
# 3. A conversation manager - ensuring the discussion stays on track and all questions are addressed
# 4. A gatekeeper - determining when a conversation is complete and can be terminated
#
# Let's create our Onboarding Coordinator agent:

# %%
# Create Coordinator Agent (will oversee the conversation)
coordinator_agent = ChatCompletionAgent(
    kernel=kernel,
    name="onboarding-coordinator",
    description="Central coordinator for the onboarding process, manages conversation flow",
    instructions="""
    You are the Onboarding Coordinator for Contoso Electronics, responsible for managing the employee onboarding experience.
    
    Your role is to:
    1. Understand the new employee's question or request
    2. Determine which specialized agent(s) should respond
    3. Synthesize information from multiple sources when necessary
    4. Ensure complete and coherent responses
    5. Provide a welcoming experience for new employees
    
    You can approve responses when they are complete and accurate. You should ask for more information
    from other agents if their responses are incomplete or unclear.
    
    Only approve the conversation when the employee's question has been fully addressed.
    """
)

# %% [markdown]
# ## Setting Up the Agent Group Chat
#
# Now that we have our specialized agents and coordinator, we need to organize them into a group chat that enables collaboration. Semantic Kernel's `AgentGroupChat` provides this capability, allowing agents to communicate with each other and work together on complex tasks.
#
# Group chats in Semantic Kernel have two key components:
#
# 1. **Termination Strategy**: Determines when a conversation is complete (when to stop agent exchanges)
# 2. **Selection Strategy**: Determines which agent should speak next (who gets control of the conversation)
#
# For our initial implementation, we'll use a simple termination strategy that lets the coordinator agent decide when the conversation is complete. Later, we'll explore more sophisticated selection strategies.

# %%
# Set up the agent group chat with the coordinator as the termination agent
group_chat = AgentGroupChat(
    agents=[hr_agent, it_agent, facilities_agent, training_agent, coordinator_agent],
    termination_strategy=DefaultTerminationStrategy(agents=[coordinator_agent])
)

# Alternative: Use a max iteration termination strategy
# group_chat = AgentGroupChat(
#     agents=[hr_agent, it_agent, facilities_agent, training_agent, coordinator_agent],
#     termination_strategy=MaxIterationTerminationStrategy(max_iterations=10)
# )

# %% [markdown]
# ## Testing the Multi-Agent System
#
# Now let's set up a function to visualize how our agents collaborate. This function will:
#
# 1. Add a user message to the group chat
# 2. Process the conversation through the group chat
# 3. Display each agent's contributions in sequence
# 4. Show the final content the user would see
#
# This allows us to observe how agents interact with each other and coordinate their responses.

# %%
from semantic_kernel.agents import AgentGroupChat
from semantic_kernel.contents import AuthorRole, ChatMessageContent

async def agent_group_chat(user_message: str) -> None:
    """Run a test scenario with the given user message."""

    await group_chat.add_chat_message(user_message)
    
    print(f"User: {user_message}\n")
    print("--- Starting Group Chat ---\n")
    
    # Process conversation and display messages
    async for response in group_chat.invoke():
                print(f"==== {response.name} just responded ====")
                print(response.content)

    content_history: list[ChatMessageContent] = []
    async for message in group_chat.get_chat_messages(agent=group_chat.agents[0]):
        if message.name == group_chat.agents[0].name:
            # The chat history contains responses from other agents.
            content_history.append(message)
    # The chat history is in descending order.
    print("Final content:")
    print(content_history[0].content)

# %% [markdown]
# ### Scenario 1: Basic Benefits Question
#
# Let's test our system with a straightforward query about health benefits. This should primarily engage the HR Policy Agent, with the coordinator managing the conversation flow. 
#
# This scenario tests:
# - Basic agent specialization (recognizing HR domain)
# - Information retrieval from plugins
# - Coordinator's ability to recognize when the question is fully answered

# %%
await agent_group_chat("What health benefits does the company offer, and how do I enroll?")

# %% [markdown]
# ### Scenario 2: Cross-Domain Query
#
# Now, let's test how the system handles a question that spans multiple domains. This represents a more complex, realistic scenario where the user has several related questions that touch different areas of expertise.
#
# This scenario tests:
# - Coordination between multiple specialized agents
# - The system's ability to handle multi-part questions
# - Information synthesis across domain boundaries

# %%
# Test with a cross-domain query
await agent_group_chat("I'm starting next Monday. What should I bring on my first day, how do I get building access, and when will I receive my laptop?")

# %% [markdown]
# ### Scenario 3: Technical Setup Question
#
# Finally, let's test a more technical query focused on development environment setup. This should primarily engage the IT Support Agent but may require input from other agents for a complete response.
#
# This scenario tests:
# - Technical domain expertise
# - The ability to provide detailed procedural instructions
# - Specialized knowledge about systems and access requirements

# %%
# Test with a technical setup question
await agent_group_chat("What's the process for setting up my development environment and getting access to the customer database?")


# %% [markdown]
# ## Advanced: Custom Selection Strategy with Built-in KernelFunctionSelectionStrategy
#
# By default, Semantic Kernel uses a `RoundRobinSelectionStrategy` that selects agents in the order they were added to the group chat. While simple, this approach doesn't account for the content of the conversation or the specialties of each agent.
#
# Semantic Kernel provides a built-in `KernelFunctionSelectionStrategy` that allows us to determine the next speaker based on the content of messages. This is more powerful than our previous approaches as it:
#
# 1. Uses the entire chat history to make decisions about who should speak next
# 2. Leverages the full power of the LLM to analyze conversation context
# 3. Integrates natively with Semantic Kernel's architecture
# 4. Allows for easy customization through prompt engineering
#
# Below, we configure a prompt-based agent selector that analyzes the conversation and chooses the most appropriate specialist to respond. This creates a more natural and coherent conversation flow, ensuring that the agent with the most relevant expertise addresses each part of the user's query.

# %%
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import KernelFunctionSelectionStrategy
from semantic_kernel.functions.kernel_arguments import KernelArguments

# Create the agent selection prompt
agent_selection_prompt = """
You are an expert at analyzing conversations and determining which specialist should respond next.

# Available Specialists:
- HR Policy Expert: Expert on company HR policies, benefits, and procedures
- IT Support Specialist: Expert on technical setup, equipment, and system access
- Facilities Coordinator: Expert on office locations, amenities, and desk assignments
- Training Specialist: Expert on required training and learning resources
- Onboarding Coordinator: Central coordinator for the onboarding process, manages conversation flow

# Topic to Agent Mapping:
- HR topics (benefits, policies, paperwork, onboarding processes): HR Policy Expert
- IT topics (computers, software, technical setup, system access): IT Support Specialist
- Facilities topics (office, building, desk, amenities): Facilities Coordinator
- Training topics (learning, courses, training, mentorship): Training Specialist

# Conversation Rules:
1. The Onboarding Coordinator handles all user queries initially
2. The Onboarding Coordinator should also respond after a specialist to ensure coherence
3. A specialist should not respond directly after themselves
4. Choose the most appropriate specialist based on the message content
5. If no specialist seems appropriate, the Onboarding Coordinator should respond

# Chat History:
{{$_history_}}

Based on the chat history, which specialist should respond next? Respond with ONLY the name of the specialist (hr-policy-agent, it-support-agent, facilities-agent, training-agent, or onboarding-coordinator) without explanation or additional text.
"""

# Add the agent selection semantic function to the kernel
agent_selector = kernel.add_function(
    plugin_name="agent_selection",
    function_name="select_next_agent", 
    prompt=agent_selection_prompt,
    description="Determines which agent should speak next based on message content"
)

# Simple result parser that just returns the trimmed text
def parse_agent_result(result):
    return str(result).strip()

# Set up the agent group chat with the coordinator as the termination agent and our new selection strategy
group_chat = AgentGroupChat(
    agents=[hr_agent, it_agent, facilities_agent, training_agent, coordinator_agent],
    termination_strategy=DefaultTerminationStrategy(agents=[coordinator_agent]),
    selection_strategy=KernelFunctionSelectionStrategy(
        kernel=kernel,
        function=agent_selector,
        result_parser=parse_agent_result
    )
)

# %% [markdown]
# ### Testing the Smart Selection Strategy: Health Benefits Question
#
# Let's test how our smarter agent selection strategy handles a benefits question. The strategy should identify that this is primarily an HR topic and direct the conversation accordingly, with the coordinator managing the flow.
#
# This test demonstrates:
# - Content-aware agent selection
# - Efficient routing of domain-specific questions
# - Contextual understanding of the conversation

# %%
await agent_group_chat("What health benefits does the company offer, and how do I enroll?")

# %% [markdown]
# ### Testing the Smart Selection Strategy: First Day Preparation
#
# This query spans multiple domains (HR, IT, and Facilities). Our selection strategy should identify the cross-domain nature and involve specialists from each relevant area to provide a comprehensive response.
#
# This test demonstrates:
# - Multi-domain question handling
# - Intelligent transitions between specialists
# - Coordinator's role in synthesizing information

# %%
await agent_group_chat("I'm starting next Monday. What should I bring on my first day, how do I get building access, and when will I receive my laptop?")

# %% [markdown]
# ### Testing the Smart Selection Strategy: Technical Setup
#
# This query is primarily IT-focused but may involve other departments for complete context. The selection strategy should identify IT Support as the primary responder while coordinating with other specialists as needed.
#
# This test demonstrates:
# - Prioritization of the most relevant specialist
# - Supplementary information from secondary specialists
# - Dynamic conversation flow based on content analysis

# %%
await agent_group_chat("What's the process for setting up my development environment and getting access to the customer database?")

# %% [markdown]
# ## Conclusion: The Power of Multi-Agent Systems
#
# In this notebook, we've implemented a sophisticated Employee Onboarding Assistant System using Semantic Kernel's Agent Framework. We demonstrated two different approaches to agent orchestration:
#
# 1. **Basic Group Chat**: Using a simple termination strategy with a coordinator agent
# 2. **Smart Selection Strategy**: Using the KernelFunctionSelectionStrategy to route conversations based on content
#
# The key advantages of this multi-agent approach include:
#
# 1. **Specialization**: Each agent focuses on their area of expertise, providing more accurate and detailed responses.
#
# 2. **Collaboration**: Agents can work together to address complex queries that span multiple domains.
#
# 3. **Coordination**: The coordinator agent ensures a coherent user experience despite the complexity behind the scenes.
#
# 4. **Extensibility**: New specialized agents can be added to the system as needs evolve.
#
# 5. **Control**: Different termination and selection strategies allow for customized conversation flows.
#
# ## Enterprise Applications
#
# This multi-agent architecture has numerous practical applications in enterprise settings:
#
# - **Customer Support**: Route inquiries to specialized support agents based on query content
# - **IT Helpdesk**: Direct technical questions to the right support specialists
# - **Knowledge Management**: Create a team of experts that can answer questions about company documentation
# - **Project Management**: Automate coordination between different project stakeholders
#
# Semantic Kernel's Agent Framework provides a robust foundation for building these types of multi-agent systems, with built-in support for group chats, termination strategies, and plugin integration.
#
# This implementation demonstrates how AI agents can collaborate to solve complex problems in enterprise settings, creating a more natural and effective user experience than what could be achieved with a single agent.

# %% [markdown]
# ## Next Steps
#
# To further enhance this system, consider:
#
# 1. Implementing more sophisticated plugins that connect to real company data sources
#
# 2. Adding memory capabilities to remember user preferences and previous interactions
#
# 3. Integrating with existing HR and IT systems for real-time information
#
# 4. Creating a web or chat interface for employees to interact with the system
#
# 5. Implementing analytics to track common questions and improve agent responses
#
# In Challenge 7-2, we'll implement a similar system using Microsoft's AutoGen framework to compare and contrast the approaches. 
