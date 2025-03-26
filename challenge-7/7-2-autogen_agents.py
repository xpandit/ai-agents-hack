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
# # Challenge 7-2: Hiring Assessment Team with AutoGen
#
# In this notebook, we'll implement a multi-agent collaboration system using Microsoft's AutoGen framework. We'll create a Hiring Assessment Team that simulates the technical assessment process for candidates, showcasing role specialization and handoffs between agents.
#
# ## What is AutoGen?
#
# AutoGen is a framework from Microsoft Research that enables:
# - Flexible agent-to-agent communication
# - Multi-agent collaboration patterns
# - Function calling and tool use
# - Dynamic conversation orchestration
#
# This implementation demonstrates how AutoGen can be used to simulate real-world business processes involving multiple specialized roles with proper handoffs.

# %% [markdown]
# ## Setting up the Environment

# %%
# Install required packages
# %pip install autogen-agentchat autogen-ext[openai,azure] python-dotenv

# %%
# Import required libraries
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Import AutoGen components
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.base import TaskResult

# Load environment variables from .env file
load_dotenv()

# %% [markdown]
# ## Configuring AutoGen with Azure OpenAI

# %%
# Check for required Azure OpenAI environment variables
required_vars = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_KEY",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION"
]

for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"Missing required environment variable: {var}")

# Configure Azure OpenAI
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.getenv("AZURE_OPENAI_KEY")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Create Azure OpenAI model client
model_client = AzureOpenAIChatCompletionClient(
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    api_version=azure_api_version,
    model="gpt-4o",
    temperature=0.3,
    timeout=300
)

print(f"Using Azure OpenAI deployment: {azure_deployment}")

# %% [markdown]
# ## Creating Tools for Our Agents
#
# Let's create some tools that our agents can use during the hiring process.

# %%
def get_job_description(role: str) -> Dict[str, Any]:
    """
    Tool to retrieve standard job descriptions for common roles.
    """
    job_descriptions = {
        "marketing_manager": {
            "title": "Marketing Manager",
            "core_skills": ["Digital marketing", "Campaign management", "Analytics", "Content strategy"],
            "experience": "5+ years in marketing with 2+ years in management",
            "education": "Bachelor's in Marketing, Business, or related field"
        },
        "software_engineer": {
            "title": "Software Engineer",
            "core_skills": ["Programming", "Problem-solving", "Version control", "Testing"],
            "experience": "3+ years of software development experience",
            "education": "CS degree or equivalent experience"
        },
        "data_scientist": {
            "title": "Data Scientist",
            "core_skills": ["Python", "Machine Learning", "Statistics", "Data visualization"],
            "experience": "2+ years working with data and analytics",
            "education": "Advanced degree in Statistics, Computer Science, or related field"
        },
        "product_manager": {
            "title": "Product Manager",
            "core_skills": ["Product strategy", "User experience", "Market research", "Agile methodologies"],
            "experience": "4+ years in product management",
            "education": "Bachelor's degree in Business, Engineering, or related field"
        }
    }
    
    if role.lower() in job_descriptions:
        return job_descriptions[role.lower()]
    else:
        return {
            "title": role.title(),
            "core_skills": ["Skill to be customized", "Skill to be customized"],
            "experience": "Experience to be customized",
            "education": "Education to be customized"
        }

def evaluate_candidate_answer(answer: str, criteria: List[str]) -> Dict[str, Any]:
    """
    Tool to evaluate a candidate's answer against assessment criteria.
    Returns a score and feedback for each criterion.
    """
    # This is a simplified implementation - in reality, this could use more sophisticated analysis
    evaluation = {
        "overall_score": 0,
        "criteria_feedback": {}
    }
    
    for criterion in criteria:
        # Simulated scoring - in reality, this would involve more analysis
        score = len([word for word in criterion.lower().split() if word in answer.lower()]) / len(criterion.split())
        score = min(score * 2, 1.0)  # Normalize to 0-1 scale
        
        evaluation["criteria_feedback"][criterion] = {
            "score": score * 10,  # Convert to 0-10 scale
            "feedback": "Candidate addressed this criterion" if score > 0.3 else "Candidate could improve on this criterion"
        }
    
    # Calculate overall score
    if evaluation["criteria_feedback"]:
        evaluation["overall_score"] = sum(item["score"] for item in evaluation["criteria_feedback"].values()) / len(evaluation["criteria_feedback"])
    
    return evaluation

# %% [markdown]
# ## Creating AutoGen Agents with Specialized Roles and Handoffs
#
# We'll create specialized agents with clear roles and the ability to handoff to each other.

# %%
# Define tool functions for agents to use
def get_interview_questions(role: str, difficulty: str = "medium") -> List[str]:
    """Generate interview questions based on role and difficulty level."""
    questions = {
        "marketing_manager": {
            "easy": [
                "What marketing channels have you used in past campaigns?",
                "How do you measure campaign success?"
            ],
            "medium": [
                "Describe a marketing campaign you led that didn't meet expectations. What did you learn?",
                "How would you allocate a $50,000 marketing budget for a new product launch?"
            ],
            "hard": [
                "How would you build a marketing strategy for a declining product with strong competition?",
                "Explain how you would implement an omnichannel marketing approach for our company."
            ]
        },
        "software_engineer": {
            "easy": [
                "What programming languages are you comfortable with?",
                "Describe your experience with version control systems."
            ],
            "medium": [
                "Explain how you would design a URL shortening service.",
                "What approaches would you take to optimize a slow-performing application?"
            ],
            "hard": [
                "Design a distributed system that can handle 10,000 transactions per second.",
                "How would you implement a real-time notification system for a global platform?"
            ]
        }
    }
    
    # Default questions if specific role not found
    default_questions = {
        "easy": [
            f"What experience do you have in {role}?",
            f"What metrics do you use to measure success in a {role} position?"
        ],
        "medium": [
            f"Describe a challenging situation you faced as a {role} and how you resolved it.",
            f"How do you stay updated with the latest trends in the {role} field?"
        ],
        "hard": [
            f"How would you implement process improvements in a {role} position?",
            f"Describe your approach to managing stakeholders with conflicting priorities."
        ]
    }
    
    # Get questions based on role and difficulty
    role_questions = questions.get(role.lower(), default_questions)
    return role_questions.get(difficulty.lower(), default_questions["medium"])

# Create Hiring Manager Agent
hiring_manager = AssistantAgent(
    name="Hiring_Manager",
    system_message="""
    You are a Hiring Manager at a growing technology company.
    
    Your role is to:
    - Define job requirements for new positions
    - Specify the skills and experience needed for the role
    - Make final hiring decisions based on assessment results
    - Delegate responsibilities to the Technical Interviewer and Recruiter

    Always start by defining the job requirements, then handoff to the Technical Interviewer 
    to create assessment questions, and then to the Recruiter to finalize the process.
    
    When you handoff to another agent, explicitly state "I'm handing this off to [agent name]" to ensure clear transitions.
    """,
    model_client=model_client,
    tools=[get_job_description],
    handoffs=["Technical_Interviewer", "Recruiter", "Review_Agent"]
)

# Create Technical Interviewer Agent
technical_interviewer = AssistantAgent(
    name="Technical_Interviewer",
    system_message="""
    You are a Technical Interviewer specializing in creating assessments for job candidates.
    
    Your role is to:
    - Design technical assessment scenarios based on job requirements
    - Create evaluation criteria for candidate responses
    - Provide expert judgment on technical competency
    - Generate specific questions to test required skills and knowledge
    
    When the Hiring Manager hands off to you, create a detailed technical assessment with clear evaluation criteria.
    Then handoff to the Recruiter to coordinate the candidate experience.
    
    When you handoff to another agent, explicitly state "I'm handing this off to [agent name]" to ensure clear transitions.
    """,
    model_client=model_client,
    tools=[get_interview_questions, evaluate_candidate_answer],
    handoffs=["Recruiter", "Hiring_Manager", "Review_Agent"]
)

# Create Recruiter Agent
recruiter = AssistantAgent(
    name="Recruiter",
    system_message="""
    You are a Recruiter specializing in candidate experience and assessment coordination.
    
    Your role is to:
    - Coordinate the overall assessment process
    - Ensure assessment scenarios are clear to candidates
    - Collect feedback from technical evaluations
    - Provide recommendations to the Hiring Manager
    - Ensure the assessment process is candidate-friendly and effective
    
    When the Technical Interviewer hands off to you, review their assessment and suggest improvements
    for candidate experience. Then provide a final recommendation and handoff back to the Hiring Manager.
    
    When you handoff to another agent, explicitly state "I'm handing this off to [agent name]" to ensure clear transitions.
    """,
    model_client=model_client,
    handoffs=["Hiring_Manager", "Technical_Interviewer", "Review_Agent"]
)

# Create Review Agent that can terminate the conversation
review_agent = AssistantAgent(
    name="Review_Agent",
    system_message="""
    You are a Review Agent responsible for evaluating the hiring assessment process created by the team.
    
    Your role is to:
    - Review the hiring assessment process for completeness and effectiveness
    - Provide final feedback on the assessment process
    - Determine when the process is ready for implementation
    - Terminate the conversation when appropriate
    
    After reviewing the process, if you find it comprehensive and ready for implementation,
    clearly state "The hiring assessment process is APPROVED" to end the conversation.
    
    If you find issues that need to be addressed, provide specific feedback and hand off to the appropriate agent.
    
    When you handoff to another agent, explicitly state "I'm handing this off to [agent name]" to ensure clear transitions.
    """,
    model_client=model_client,
    handoffs=["Hiring_Manager", "Technical_Interviewer", "Recruiter"]
)

# %% [markdown]
# ## Setting Up the Group Chat with Swarm Pattern
#
# We'll create a group chat with a swarm pattern to enable specialized collaboration between the agents.

# %%
# Create a conversation termination condition
termination_condition = TextMentionTermination("APPROVED")

# Create a group chat with all agents
team_chat = RoundRobinGroupChat(
    [hiring_manager, technical_interviewer, recruiter, review_agent],
    termination_condition=termination_condition,
    max_turns=20  # Increase maximum conversation rounds to accommodate the new agent
)

# %% [markdown]
# ## Running the Hiring Assessment Scenario
#
# Let's run our scenario where the Hiring Manager initiates the process for creating a technical assessment.

# %%
# Define the scenario request
initial_request = """
We need to create a hiring assessment for a Data Scientist position. The candidate should have strong skills in
machine learning, Python programming, and data visualization. I want a comprehensive technical assessment that
evaluates both their technical knowledge and problem-solving abilities.

Please work together to create an appropriate assessment process, including specific questions, evaluation criteria,
and a candidate-friendly approach.
"""

# %%
async def run_hiring_assessment_scenario(initial_request: str):
    print(f"Initial Request: {initial_request}\n")
    print("--- Starting Hiring Assessment Scenario ---\n")
    
    # Start the conversation with the simulation manager
    stream = team_chat.run_stream(task=initial_request)
    
    async for message in stream:
        if isinstance(message, TaskResult):
            print("Stop Reason:", message.stop_reason)
        else:
            # Format the message for better readability
            if hasattr(message, 'source') and hasattr(message, 'content'):
                # Format agent name
                agent_name = message.source
                
                # Handle different message types for cleaner output
                if str(type(message)).endswith("TextMessage'>"):
                    print(f"📝 {agent_name}:\n{message.content}\n")
                
                elif str(type(message)).endswith("HandoffMessage'>"):
                    print(f"🔄 {agent_name} hands off to {message.target}\n")
                
                elif str(type(message)).endswith("ToolCallSummaryMessage'>"):
                    print(f"🛠️ {agent_name} used tool with result:\n{message.content}\n")
                
                # Skip technical messages that clutter the output
                elif str(type(message)).endswith("ToolCallRequestEvent'>") or str(type(message)).endswith("ToolCallExecutionEvent'>"):
                    pass
                
                else:
                    # Fallback for other message types
                    print(f"{agent_name}: {message.content}")
    
    print("\n--- Scenario Complete ---\n")
    return stream

# %%
await run_hiring_assessment_scenario("We need to hire a Product Manager for our new AI initiative. Please create a detailed hiring assessment process.")

# %% [markdown]
# ## Conclusion
#
# This implementation demonstrates the power of AutoGen for multi-agent collaboration. The key features showcased include:
#
# 1. **Specialized Agent Roles** - Each agent has a distinct role and expertise
# 2. **Agent Handoffs** - Clear transitions between agents with explicit handoff patterns
# 3. **Tool Integration** - Specialized tools for job descriptions, questions, and evaluation
# 4. **Realistic Simulation** - Practical assessment creation process that mimics real-world hiring teams
#
# The resulting system provides a realistic simulation of how hiring teams collaborate to create effective assessments for job candidates, with each agent contributing their specialized expertise and clearly defined handoffs between roles.
