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
# # Challenge 7-2: Multi-Agent Collaboration with AutoGen
#
# In this notebook, we'll implement a multi-agent collaboration system using Microsoft's AutoGen framework. Similar to our Semantic Kernel implementation, we'll create an Employee Onboarding Assistant System but this time using AutoGen's flexible agent communication patterns.
#
# ## What is AutoGen?
#
# AutoGen is a research-oriented framework maintained by Microsoft Research's AI Frontiers Lab, specializing in:
#
# - Advanced conversable agent design patterns
# - Flexible agent-to-agent communication protocols
# - Dynamic group chat orchestration with various topologies
# - Code execution and tool use within agent conversations
# - Cutting-edge research on agentic capabilities
# - Highly customizable agent behaviors and interactions
#
# AutoGen focuses on enabling autonomous and collaborative problem-solving between multiple AI agents. The framework allows these agents to work together through conversation, execute code, use tools, and interact with humans when needed.

# %% [markdown]
# ## Setting up the Environment
#
# First, let's install the necessary packages and set up our environment.

# %%
# Install required packages
# !pip install autogen-agentchat autogen-ext[openai] python-dotenv

# %%
# Import required libraries
import os
import json
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Import AutoGen components
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, config_list_from_json

# Load environment variables from .env file
load_dotenv()

# %% [markdown]
# ## Configuring AutoGen
#
# Let's configure AutoGen with our API settings. For this example, we'll use OpenAI's GPT-4 model, but you could also use other compatible models.

# %%
# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

# Configure the OpenAI API
config_list = [
    {
        "model": os.getenv("OPENAI_MODEL_NAME", "gpt-4"),
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
]

# Create LLM configuration for agents
llm_config = {
    "config_list": config_list,
    "temperature": 0.3,
    "timeout": 300,
}

print(f"Using model: {config_list[0]['model']}")

# %% [markdown]
# ## Creating the Knowledge Base
#
# For our Employee Onboarding Assistant System, we need to provide our agents with information about company policies, procedures, and resources. In real-world applications, this would typically be fetched from company databases or knowledge management systems. For this example, we'll create a simple knowledge base as a Python dictionary.

# %%
# Define a knowledge base with company information
knowledge_base = {
    "hr_policies": {
        "benefits": {
            "health": "Contoso offers comprehensive health insurance with dental and vision coverage.",
            "retirement": "Employees can enroll in the 401(k) plan with 5% company matching.",
            "pto": "New employees start with 15 days of PTO plus 10 holidays annually.",
            "parental": "Contoso provides 12 weeks of paid parental leave for all new parents."
        },
        "onboarding": [
            "Complete I-9 and tax forms",
            "Enroll in benefits plans (within 30 days)",
            "Set up direct deposit for payroll",
            "Complete required compliance training",
            "Schedule orientation session with HR",
            "Review employee handbook"
        ],
        "policies": {
            "remote_work": "Contoso has a hybrid work policy allowing 3 days of remote work per week.",
            "dress_code": "Business casual dress code is standard, with casual Fridays.",
            "expenses": "All business expenses require manager approval and receipts.",
            "travel": "Travel must be booked through the company travel portal."
        }
    },
    "it_support": {
        "equipment": {
            "laptop": "Standard laptops are Dell XPS 13 or MacBook Pro 14\".",
            "monitors": "Each employee receives two 27\" 4K monitors.",
            "accessories": "Standard accessories include keyboard, mouse, headset, and docking station.",
            "mobile": "Employees may choose between iPhone or Android devices."
        },
        "software": [
            "Microsoft 365 (including email, Teams, OneDrive)",
            "Slack for team communication",
            "Jira for project management",
            "GitHub for code repositories",
            "Zoom for video conferencing"
        ],
        "setup": {
            "windows": """
            Windows Setup Instructions:
            1. Power on your Windows laptop and login with provided credentials
            2. Connect to the company WiFi: "Contoso-Secure" (password will be provided)
            3. Launch the IT Setup application from the desktop
            4. Follow the prompts to install required software and security tools
            5. Set up VPN using GlobalProtect (instructions in IT portal)
            """,
            "mac": """
            Mac Setup Instructions:
            1. Power on your Mac and follow initial setup
            2. Connect to the company WiFi: "Contoso-Secure" (password will be provided)
            3. Install company certificate from IT portal
            4. Set up VPN using GlobalProtect (instructions in IT portal)
            5. Login to company portal with your credentials to install required software
            """
        }
    },
    "facilities": {
        "locations": {
            "Seattle": "Main HQ: 123 Tech Blvd, Seattle, WA",
            "San Francisco": "West Coast Office: 456 Innovation St, San Francisco, CA",
            "New York": "East Coast Office: 789 Enterprise Ave, New York, NY",
            "London": "European HQ: 101 Digital Ln, London, UK"
        },
        "amenities": {
            "Seattle": ["On-site gym", "Cafeteria", "Game room", "Rooftop lounge", "Bike storage"],
            "San Francisco": ["Coffee bar", "Meditation room", "Gym discount", "Rooftop garden"],
            "New York": ["On-site cafe", "Fitness center", "Library", "Meditation room"],
            "London": ["Tea room", "Fitness center", "Cafeteria", "Game area"]
        },
        "desk_assignment": """
        Desk Assignment Process:
        1. Your team lead will show you to your assigned desk on your first day
        2. Your desk will be labeled with your name
        3. Standard equipment will be set up and waiting for you
        4. If you need any special accommodations, please notify HR in advance
        5. Hot-desking is available in designated areas for flexible work
        """
    },
    "training": {
        "required": {
            "all_employees": [
                "Security and Compliance (all employees)",
                "Anti-Harassment Policy (all employees)",
                "Company Values and Culture (all employees)"
            ],
            "engineering": ["Secure Coding Practices", "Code Review Process", "DevOps Pipeline Training"],
            "sales": ["CRM System Training", "Sales Methodology", "Product Knowledge Certification"],
            "hr": ["HRIS System Training", "Employment Law Basics", "Benefits Administration"],
            "marketing": ["Marketing Platform Training", "Brand Guidelines", "Analytics Tools"]
        },
        "resources": [
            "Contoso Learning Portal: https://learn.contoso.com",
            "LinkedIn Learning (free access for employees)",
            "Department-specific documentation libraries",
            "Weekly lunch-and-learn sessions",
            "Quarterly development workshops"
        ],
        "mentor_program": """
        Contoso Mentor Program:
        - Each new employee is paired with an experienced mentor
        - Mentors provide guidance, answer questions, and help with networking
        - Mentor relationships typically last 3-6 months
        - Mentor assignments are made during your first week
        - Monthly check-ins are encouraged to track progress
        """
    }
}

# Save the knowledge base to a file (for agent code execution)
knowledge_base_path = Path("knowledge_base.json")
with open(knowledge_base_path, "w") as f:
    json.dump(knowledge_base, f, indent=2)

print(f"Knowledge base saved to {knowledge_base_path}")

# %% [markdown]
# ## Creating Custom Functions for Agents
#
# In AutoGen, we can define functions that agents can call to retrieve specific information. This allows our agents to have structured access to the knowledge base.

# %%
# Create custom functions to access the knowledge base
def get_hr_policy_info(policy_type: Optional[str] = None) -> str:
    """
    Retrieve information about HR policies.
    
    Args:
        policy_type: Optional specific policy to retrieve
    
    Returns:
        Policy information as a string
    """
    
    policies = knowledge_base["hr_policies"]["policies"]
    
    if policy_type and policy_type.lower().replace(" ", "_") in policies:
        return policies[policy_type.lower().replace(" ", "_")]
    else:
        return "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in policies.items()])

def get_benefits_info(benefit_type: Optional[str] = None) -> str:
    """
    Retrieve information about employee benefits.
    
    Args:
        benefit_type: Optional specific benefit to retrieve
    
    Returns:
        Benefit information as a string
    """
    
    benefits = knowledge_base["hr_policies"]["benefits"]
    
    if benefit_type and benefit_type.lower() in benefits:
        return benefits[benefit_type.lower()]
    else:
        return "\n".join([f"- {k.capitalize()}: {v}" for k, v in benefits.items()])

def get_onboarding_checklist() -> str:
    """
    Retrieve the onboarding checklist for new employees.
    
    Returns:
        Onboarding checklist as a string
    """
    
    checklist = knowledge_base["hr_policies"]["onboarding"]
    return "\n".join([f"- {item}" for item in checklist])

def get_equipment_info(equipment_type: Optional[str] = None) -> str:
    """
    Retrieve information about standard equipment.
    
    Args:
        equipment_type: Optional specific equipment type to retrieve
    
    Returns:
        Equipment information as a string
    """
    
    equipment = knowledge_base["it_support"]["equipment"]
    
    if equipment_type and equipment_type.lower() in equipment:
        return equipment[equipment_type.lower()]
    else:
        return "\n".join([f"- {k.capitalize()}: {v}" for k, v in equipment.items()])

def get_software_access() -> str:
    """
    Retrieve information about standard software and access.
    
    Returns:
        Software information as a string
    """
    
    software = knowledge_base["it_support"]["software"]
    return "\n".join([f"- {item}" for item in software])

def get_setup_instructions(system_type: str = "windows") -> str:
    """
    Retrieve setup instructions for company systems.
    
    Args:
        system_type: Type of system ("windows" or "mac")
    
    Returns:
        Setup instructions as a string
    """
    
    setup = knowledge_base["it_support"]["setup"]
    return setup.get(system_type.lower(), setup["windows"])

def get_office_locations() -> str:
    """
    Retrieve information about office locations.
    
    Returns:
        Office locations as a string
    """
    
    locations = knowledge_base["facilities"]["locations"]
    return "\n".join([f"- {k}: {v}" for k, v in locations.items()])

def get_office_amenities(location: Optional[str] = None) -> str:
    """
    Retrieve information about office amenities.
    
    Args:
        location: Optional specific location to retrieve amenities for
    
    Returns:
        Amenities information as a string
    """
    
    amenities = knowledge_base["facilities"]["amenities"]
    
    if location and location.title() in amenities:
        return "\n".join([f"- {item}" for item in amenities[location.title()]])
    else:
        return "All Contoso offices feature modern workspaces, break areas, meeting rooms, and video conferencing facilities."

def get_desk_assignment_process() -> str:
    """
    Retrieve information about desk assignment process.
    
    Returns:
        Desk assignment process as a string
    """
    
    return knowledge_base["facilities"]["desk_assignment"]

def get_required_training(department: Optional[str] = None) -> str:
    """
    Retrieve information about required training.
    
    Args:
        department: Optional specific department to retrieve training for
    
    Returns:
        Training information as a string
    """
    
    general_training = knowledge_base["training"]["required"]["all_employees"]
    department_specific = knowledge_base["training"]["required"]
    
    if department and department.lower() in department_specific:
        result = "Required training:\n"
        result += "\n".join([f"- {item}" for item in general_training])
        result += "\n\nDepartment-specific training:\n"
        result += "\n".join([f"- {item}" for item in department_specific[department.lower()]])
        return result
    else:
        return "Required training for all employees:\n" + "\n".join([f"- {item}" for item in general_training])

def get_learning_resources() -> str:
    """
    Retrieve information about learning resources.
    
    Returns:
        Learning resources as a string
    """
    
    resources = knowledge_base["training"]["resources"]
    return "\n".join([f"- {item}" for item in resources])

def get_mentor_program_info() -> str:
    """
    Retrieve information about the mentor program.
    
    Returns:
        Mentor program information as a string
    """
    
    return knowledge_base["training"]["mentor_program"]

# %% [markdown]
# ## Creating Specialized Agents
#
# Now, let's create the specialized agents that will be part of our Employee Onboarding Assistant System. In AutoGen, we'll use `AssistantAgent` for our specialized agents and `UserProxyAgent` for the user interface.

# %%
# Define the function descriptions for function calling
function_map = {
    "get_hr_policy_info": {
        "name": "get_hr_policy_info",
        "description": "Get information about HR policies",
        "parameters": {
            "type": "object",
            "properties": {
                "policy_type": {
                    "type": "string",
                    "description": "The specific policy to retrieve information about (remote work, dress code, expenses, travel)",
                },
            },
            "required": [],
        },
    },
    "get_benefits_info": {
        "name": "get_benefits_info",
        "description": "Get information about employee benefits",
        "parameters": {
            "type": "object",
            "properties": {
                "benefit_type": {
                    "type": "string",
                    "description": "The specific benefit to retrieve information about (health, retirement, pto, parental)",
                },
            },
            "required": [],
        },
    },
    "get_onboarding_checklist": {
        "name": "get_onboarding_checklist",
        "description": "Get the onboarding checklist for new employees",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "get_equipment_info": {
        "name": "get_equipment_info",
        "description": "Get information about standard equipment provided to employees",
        "parameters": {
            "type": "object",
            "properties": {
                "equipment_type": {
                    "type": "string",
                    "description": "The specific equipment to retrieve information about (laptop, monitors, accessories, mobile)",
                },
            },
            "required": [],
        },
    },
    "get_software_access": {
        "name": "get_software_access",
        "description": "Get information about standard software and access",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "get_setup_instructions": {
        "name": "get_setup_instructions",
        "description": "Get setup instructions for company systems",
        "parameters": {
            "type": "object",
            "properties": {
                "system_type": {
                    "type": "string",
                    "description": "The type of system (windows or mac)",
                },
            },
            "required": [],
        },
    },
    "get_office_locations": {
        "name": "get_office_locations",
        "description": "Get information about office locations",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "get_office_amenities": {
        "name": "get_office_amenities",
        "description": "Get information about office amenities",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The specific location to retrieve amenities for (Seattle, San Francisco, New York, London)",
                },
            },
            "required": [],
        },
    },
    "get_desk_assignment_process": {
        "name": "get_desk_assignment_process",
        "description": "Get information about desk assignment process",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "get_required_training": {
        "name": "get_required_training",
        "description": "Get information about required training",
        "parameters": {
            "type": "object",
            "properties": {
                "department": {
                    "type": "string",
                    "description": "The specific department to retrieve training for (engineering, sales, hr, marketing)",
                },
            },
            "required": [],
        },
    },
    "get_learning_resources": {
        "name": "get_learning_resources",
        "description": "Get information about learning resources",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "get_mentor_program_info": {
        "name": "get_mentor_program_info",
        "description": "Get information about the mentor program",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

# Create the function mapping for execution
function_executor_mapping = {
    "get_hr_policy_info": get_hr_policy_info,
    "get_benefits_info": get_benefits_info,
    "get_onboarding_checklist": get_onboarding_checklist,
    "get_equipment_info": get_equipment_info,
    "get_software_access": get_software_access,
    "get_setup_instructions": get_setup_instructions,
    "get_office_locations": get_office_locations,
    "get_office_amenities": get_office_amenities,
    "get_desk_assignment_process": get_desk_assignment_process,
    "get_required_training": get_required_training,
    "get_learning_resources": get_learning_resources,
    "get_mentor_program_info": get_mentor_program_info,
}

# %% [markdown]
# ## Creating AutoGen Agents
#
# Now let's create our specialized agents using AutoGen's AssistantAgent class. Each agent will be given a specific role and access to the relevant functions.

# %%
# Create HR Policy Agent
hr_agent = AssistantAgent(
    name="HR_Policy_Expert",
    system_message="""
    You are an HR Policy Expert for Contoso Electronics.
    
    Your role is to help new employees understand:
    - Company benefits and how to enroll
    - HR policies and procedures
    - Required documentation and forms
    - Onboarding processes and checklists
    
    Use the available functions to retrieve accurate information from the company knowledge base.
    If asked about topics outside your expertise, defer to the appropriate specialized agent.
    
    Always be helpful, informative, and welcoming to new employees.
    """,
    llm_config={
        "config_list": config_list,
        "temperature": 0.3,
        "functions": [
            function_map["get_hr_policy_info"],
            function_map["get_benefits_info"],
            function_map["get_onboarding_checklist"],
        ],
        "function_call": "auto",
    }
)

# Create IT Support Agent
it_agent = AssistantAgent(
    name="IT_Support_Specialist",
    system_message="""
    You are an IT Support Specialist for Contoso Electronics.
    
    Your role is to help new employees with:
    - Setting up their computers and equipment
    - Accessing company systems and software
    - Understanding IT policies and security requirements
    - Troubleshooting common technical issues
    
    Use the available functions to retrieve accurate information from the company knowledge base.
    If asked about topics outside your expertise, defer to the appropriate specialized agent.
    
    Provide clear, step-by-step instructions and be patient with technical questions.
    """,
    llm_config={
        "config_list": config_list,
        "temperature": 0.3,
        "functions": [
            function_map["get_equipment_info"],
            function_map["get_software_access"],
            function_map["get_setup_instructions"],
        ],
        "function_call": "auto",
    }
)

# Create Facilities Agent
facilities_agent = AssistantAgent(
    name="Facilities_Coordinator",
    system_message="""
    You are a Facilities Coordinator for Contoso Electronics.
    
    Your role is to help new employees with:
    - Office locations and layouts
    - Building access and security
    - Desk assignments and workspaces
    - Office amenities and resources
    - Parking and transportation options
    
    Use the available functions to retrieve accurate information from the company knowledge base.
    If asked about topics outside your expertise, defer to the appropriate specialized agent.
    
    Focus on making new employees feel comfortable in their physical work environment.
    """,
    llm_config={
        "config_list": config_list,
        "temperature": 0.3,
        "functions": [
            function_map["get_office_locations"],
            function_map["get_office_amenities"],
            function_map["get_desk_assignment_process"],
        ],
        "function_call": "auto",
    }
)

# Create Training Agent
training_agent = AssistantAgent(
    name="Training_Specialist",
    system_message="""
    You are a Training and Development Specialist for Contoso Electronics.
    
    Your role is to help new employees with:
    - Required onboarding training modules
    - Learning resources and platforms
    - Professional development opportunities
    - Mentorship programs
    - Department-specific training
    
    Use the available functions to retrieve accurate information from the company knowledge base.
    If asked about topics outside your expertise, defer to the appropriate specialized agent.
    
    Be encouraging and supportive of employees' learning journeys.
    """,
    llm_config={
        "config_list": config_list,
        "temperature": 0.3,
        "functions": [
            function_map["get_required_training"],
            function_map["get_learning_resources"],
            function_map["get_mentor_program_info"],
        ],
        "function_call": "auto",
    }
)

# Create Coordinator Agent (will oversee the conversation)
coordinator_agent = AssistantAgent(
    name="Onboarding_Coordinator",
    system_message="""
    You are the Onboarding Coordinator for Contoso Electronics, responsible for managing the employee onboarding experience.
    
    Your role is to:
    1. Understand the new employee's question or request
    2. Determine which specialized agent(s) should respond
    3. Synthesize information from multiple sources when necessary
    4. Ensure complete and coherent responses
    5. Provide a welcoming experience for new employees
    
    You can approve responses when they are complete and accurate. You should ask for more information
    from other agents if their responses are incomplete or unclear.
    
    Only indicate the conversation is complete when the employee's question has been fully addressed.
    """,
    llm_config={
        "config_list": config_list,
        "temperature": 0.3,
    }
)

# Create User Proxy Agent for function execution
user_proxy = UserProxyAgent(
    name="New_Employee",
    human_input_mode="NEVER",
    function_map=function_executor_mapping,
    code_execution_config=False,
    system_message="""
    You represent a new employee asking onboarding questions.
    You will execute functions on behalf of the agents.
    """
)

# %% [markdown]
# ## Setting Up the Group Chat
#
# Now we'll configure a Group Chat with all our specialized agents to enable multi-agent collaboration.

# %%
# Create a group chat with all agents
groupchat = GroupChat(
    agents=[user_proxy, coordinator_agent, hr_agent, it_agent, facilities_agent, training_agent],
    messages=[],
    max_round=15  # Maximum conversation rounds
)

# Create a manager to handle the conversation flow
manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)

# %% [markdown]
# ## Testing the Multi-Agent System
#
# Let's test our Employee Onboarding Assistant System with various scenarios.

# %% [markdown]
# ### Scenario 1: Basic Benefits Question

# %%
# Helper function to run scenarios and display results
def run_scenario(question):
    # Reset all agents for a fresh conversation
    for agent in groupchat.agents:
        agent.reset()
    
    # Reset the group chat messages
    groupchat.messages = []
    
    print(f"New Employee: {question}\n")
    print("--- Starting Group Chat ---\n")
    
    # Initiate the conversation with a query
    user_proxy.initiate_chat(
        manager,
        message=question
    )
    
    print("\n--- Chat Complete ---\n")

# %%
# Test with a basic benefits question
run_scenario("What health benefits does the company offer, and how do I enroll?")

# %% [markdown]
# ### Scenario 2: Cross-Domain Query
#
# Let's test how the system handles a question that spans multiple domains.

# %%
# Test with a cross-domain query
run_scenario("I'm starting next Monday. What should I bring on my first day, how do I get building access, and when will I receive my laptop?")

# %% [markdown]
# ### Scenario 3: Technical Setup Question

# %%
# Test with a technical setup question
run_scenario("What's the process for setting up my Mac development environment and getting access to the company systems?")

# %% [markdown]
# ### Scenario 4: Policy Clarification

# %%
# Test with a policy clarification question
run_scenario("What's the company policy on remote work, and how do I request equipment for my home office?")

# %% [markdown]
# ### Scenario 5: Complex Multi-Step Process

# %%
# Test with a complex multi-step process
run_scenario("I need to set up direct deposit for my paycheck, choose my health insurance plan, and schedule my orientation. Where do I start?")

# %% [markdown]
# ## Customizing the Group Chat Flow
#
# One of AutoGen's strengths is its flexibility in configuring agent interactions. Let's experiment with different group chat configurations to customize the conversation flow.

# %% [markdown]
# ### Custom Speaker Selection

# %%
# Define a custom speaker selection function
def select_next_speaker(last_speaker, groupchat):
    """
    Custom function to determine the next speaker in the group chat.
    
    Args:
        last_speaker: The agent who spoke last
        groupchat: The group chat instance
    
    Returns:
        The name of the next agent to speak
    """
    # If the user just spoke, the coordinator should respond first
    if last_speaker.name == "New_Employee":
        return "Onboarding_Coordinator"
    
    # If the coordinator just spoke, look at the message to see who should respond
    if last_speaker.name == "Onboarding_Coordinator":
        last_message = groupchat.messages[-1]["content"].lower()
        
        # Check message content to route to the appropriate specialized agent
        if any(keyword in last_message for keyword in ["benefit", "policy", "policies", "hr", "onboarding"]):
            return "HR_Policy_Expert"
        elif any(keyword in last_message for keyword in ["computer", "laptop", "equipment", "software", "technical", "setup"]):
            return "IT_Support_Specialist"
        elif any(keyword in last_message for keyword in ["office", "building", "desk", "location", "parking"]):
            return "Facilities_Coordinator"
        elif any(keyword in last_message for keyword in ["training", "learning", "course", "mentor"]):
            return "Training_Specialist"
        else:
            # Default to a round-robin approach if no keywords match
            agents = [a.name for a in groupchat.agents if a.name not in ["New_Employee", "Onboarding_Coordinator", last_speaker.name]]
            return agents[0] if agents else "Onboarding_Coordinator"
    
    # For other cases, the coordinator should generally speak next to maintain flow
    return "Onboarding_Coordinator"

# Create a new group chat with custom speaker selection
custom_groupchat = GroupChat(
    agents=[user_proxy, coordinator_agent, hr_agent, it_agent, facilities_agent, training_agent],
    messages=[],
    max_round=15,
    speaker_selection_method=select_next_speaker
)

# Create a new manager with the custom group chat
custom_manager = GroupChatManager(
    groupchat=custom_groupchat,
    llm_config=llm_config,
)

# %% [markdown]
# ### Test Custom Speaker Selection

# %%
# Helper function to run scenarios with the custom group chat
def run_custom_scenario(question):
    # Reset all agents for a fresh conversation
    for agent in custom_groupchat.agents:
        agent.reset()
    
    # Reset the group chat messages
    custom_groupchat.messages = []
    
    print(f"New Employee: {question}\n")
    print("--- Starting Custom Group Chat ---\n")
    
    # Initiate the conversation with a query
    user_proxy.initiate_chat(
        custom_manager,
        message=question
    )
    
    print("\n--- Chat Complete ---\n")

# %%
# Test with a cross-domain query using custom speaker selection
run_custom_scenario("I'm starting next week in the engineering department. What training do I need to complete, when will I get my laptop, and how do I set up direct deposit for my paycheck?")

# %% [markdown]
# ## Advanced: Implementing a Collaborative Problem-Solving Workflow
#
# Let's implement a more advanced workflow where agents collaborate to solve a complex employee onboarding problem. This demonstrates how AutoGen can be used to create sophisticated multi-agent workflows.

# %%
# Define a Team Lead agent for a more complex scenario
team_lead_agent = AssistantAgent(
    name="Team_Lead",
    system_message="""
    You are a Team Lead at Contoso Electronics. You provide guidance on team-specific onboarding processes.
    
    Your role includes:
    - Explaining team workflows and conventions
    - Setting expectations for new team members
    - Helping new employees integrate into the team
    - Providing mentorship and guidance
    - Explaining project-specific requirements
    
    Focus on helping new team members become productive contributors quickly.
    """,
    llm_config={
        "config_list": config_list,
        "temperature": 0.3,
    }
)

# Define a Project Agent for the advanced workflow
project_agent = AssistantAgent(
    name="Project_Manager",
    system_message="""
    You are a Project Manager at Contoso Electronics. You provide information about current projects.
    
    Your role includes:
    - Explaining project roadmaps and timelines
    - Clarifying project roles and responsibilities
    - Sharing information about project tools and resources
    - Helping new employees understand project priorities
    - Facilitating project onboarding activities
    
    Focus on helping new team members understand the project landscape and how they can contribute.
    """,
    llm_config={
        "config_list": config_list,
        "temperature": 0.3,
    }
)

# %% [markdown]
# ## Problem-Solving Workflow with Specialized Roles
#
# Let's create a special workflow for a new engineer who needs to get fully set up to contribute to a project. This involves multiple specialized agents coordinating to solve a complex onboarding scenario.

# %%
# Create an enhanced group chat for the complex workflow
workflow_groupchat = GroupChat(
    agents=[user_proxy, coordinator_agent, hr_agent, it_agent, facilities_agent, training_agent, team_lead_agent, project_agent],
    messages=[],
    max_round=20
)

workflow_manager = GroupChatManager(
    groupchat=workflow_groupchat,
    llm_config=llm_config,
)

# %%
# Run the complex workflow scenario
def run_workflow_scenario():
    # Reset all agents for a fresh conversation
    for agent in workflow_groupchat.agents:
        agent.reset()
    
    # Reset the group chat messages
    workflow_groupchat.messages = []
    
    # Complex onboarding scenario for a new software engineer
    scenario = """
    I'm a new software engineer starting next Monday on the cloud infrastructure team. I need to:
    1. Complete all required onboarding paperwork and training
    2. Set up my development environment with access to the right repositories
    3. Understand the current project I'll be working on
    4. Connect with my team members and mentor
    5. Find my desk and learn about office resources
    
    Can you create a complete plan for my first week, with a day-by-day schedule to make sure I get everything done efficiently?
    """
    
    print(f"New Software Engineer: {scenario}\n")
    print("--- Starting Complex Workflow Group Chat ---\n")
    
    # Initiate the conversation with the scenario
    user_proxy.initiate_chat(
        workflow_manager,
        message=scenario
    )
    
    print("\n--- Workflow Complete ---\n")

# %%
# Run the complex workflow scenario
run_workflow_scenario()

# %% [markdown]
# ## Conclusion: The Power of Multi-Agent Systems with AutoGen
#
# In this notebook, we've implemented a sophisticated Employee Onboarding Assistant System using AutoGen's flexible multi-agent framework. The key advantages of this approach include:
#
# 1. **Flexible Agent Communication**: AutoGen's GroupChat allows for dynamic interaction patterns between agents.
#
# 2. **Function Calling**: Agents can access structured information through function calling.
#
# 3. **Custom Speaker Selection**: The ability to control conversation flow with custom speaker selection logic.
#
# 4. **Scalable Agent Teams**: Easy addition of new specialized agents to handle broader domains.
#
# 5. **Complex Workflows**: Support for sophisticated multi-step workflows spanning multiple domains.
#
# The AutoGen framework is particularly well-suited for research and development of advanced agent interaction patterns, with its focus on flexible, collaborative problem-solving between agents.
#
# This implementation demonstrates how a team of AI agents can work together to address complex onboarding scenarios, providing a coherent and comprehensive user experience that draws on specialized knowledge across multiple domains.

# %% [markdown]
# ## Comparing Semantic Kernel and AutoGen Approaches
#
# Having implemented multi-agent systems using both Semantic Kernel (in Challenge 7-1) and AutoGen (in this notebook), we can compare the two approaches:
#
# | Feature | Semantic Kernel | AutoGen |
# | --- | --- | --- |
# | **Production Readiness** | Designed for enterprise production use | Research-oriented framework |
# | **Agent Definition** | Structured with ChatCompletionAgent | Flexible with AssistantAgent and UserProxyAgent |
# | **Group Chat** | AgentGroupChat with built-in termination strategies | GroupChat with customizable speaker selection |
# | **Function Integration** | Plugin architecture (object-oriented) | Function calling (dictionary-based) |
# | **Conversation Flow** | Selection and termination strategies | Speaker selection methods and round-based termination |
# | **Code Execution** | Less integrated | Strong code execution capabilities |
# | **Development Focus** | Enterprise integration | Research and experimentation |
#
# Both frameworks have their strengths and are converging in capabilities over time. The choice between them depends on your specific requirements:
#
# - Choose **Semantic Kernel** for enterprise-grade applications with structured workflows and strong plugin architecture
# - Choose **AutoGen** for research-oriented applications, flexible agent interactions, and code execution capabilities
#
# In the future, these frameworks are expected to converge further, offering the best of both worlds for AI agent development.

# %% [markdown]
# ## Next Steps
#
# To further enhance this system, consider:
#
# 1. Implementing persistent memory to track user preferences and history
#
# 2. Adding document retrieval capabilities for company documentation
#
# 3. Connecting to real HR and IT systems through APIs
#
# 4. Creating a web or chat interface for employees to interact with the system
#
# 5. Implementing user feedback mechanisms to improve agent responses over time 
