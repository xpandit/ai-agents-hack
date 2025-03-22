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
# # Challenge 5: Azure AI Agents SDK - Employee Onboarding & IT Support
#
# In this notebook, we'll explore the Azure AI Agents SDK through a series of focused examples. Unlike the previous notebook that used Semantic Kernel, this one will leverage the Azure AI Agents SDK to create AI assistants with various capabilities.
#
# The examples all revolve around employee onboarding and IT support, but each one focuses on a different capability of the Azure AI Agents SDK. You can run them independently and modify them to experiment with different aspects of the SDK.
#
# ## What is Azure AI Agents SDK?
#
# The Azure AI Agents SDK is a client library that allows you to build and run AI agents in Azure. It provides capabilities to:
#
# - Create agents that can execute tasks and answer questions
# - Add tools like file search, code interpreter, and custom functions
# - Set up conversational threads to maintain context
# - Process runs to get agent responses
# - Track detailed run steps to understand agent behavior

# %% [markdown]
# ## Setting up the Environment
#
# First, let's install the necessary packages and set up our environment.

# %%
# !pip install azure-ai-projects azure-identity openai python-dotenv PyPDF2

# %%
import os
import json
import uuid
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, TypedDict, Annotated

import asyncio
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

# Import the Azure AI Projects SDK components
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    CodeInterpreterTool,
    FunctionTool,
    FileSearchTool,
    JsonSchemaType,
    JsonObject,
    UserFunction,
    FunctionParameter,
    ToolSet
)

# Load environment variables
load_dotenv()

# Azure AI Project configuration
project_connection_string = os.getenv("PROJECT_CONNECTION_STRING")
model_deployment_name = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4o-mini")

# Initialize the AI Project client
project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=project_connection_string,
)

# %% [markdown]
# ## Example 1: Basic Agent for Employee Onboarding
#
# In this first example, we'll create a simple agent that can answer basic questions about employee onboarding without any special tools.

# %%
async def example1_basic_agent():
    """
    This example demonstrates how to create a simple agent without special tools
    and how to interact with it through a conversation thread.
    """
    print("Running Example 1: Basic Agent for Employee Onboarding")
    
    # Create a basic agent
    agent = await project_client.agents.create_agent_async(
        model=model_deployment_name,
        name="Basic Onboarding Assistant",
        instructions="""
        You are an Employee Onboarding Assistant designed to help new employees 
        get familiar with company policies and procedures. 
        
        Answer questions about:
        - First day procedures
        - HR policies
        - Company culture
        - Office locations
        - IT setup
        
        Always be helpful, concise, and welcoming to new employees.
        """
    )
    
    print(f"Created agent, ID: {agent.id}")
    
    # Create a thread for the conversation
    thread = await project_client.agents.create_thread_async()
    print(f"Created thread, ID: {thread.id}")
    
    # Add a message to the thread
    message = await project_client.agents.create_message_async(
        thread_id=thread.id,
        role="user",
        content="Hi! I'm a new employee starting next week. What should I expect on my first day?"
    )
    print(f"Created message, ID: {message.id}")
    
    # Create and process a run
    run = await project_client.agents.create_and_process_run_async(
        thread_id=thread.id, 
        assistant_id=agent.id
    )
    print(f"Run finished with status: {run.status}")
    
    if run.status == "failed":
        print(f"Run failed: {run.last_error}")
    
    # Get all messages in the thread
    messages = await project_client.agents.list_messages_async(thread_id=thread.id)
    
    # Display the assistant's response
    assistant_messages = [m for m in messages if m.role == "assistant"]
    if assistant_messages:
        print("\nAssistant's response:")
        print(assistant_messages[-1].content[0].text)
    
    # Ask a follow-up question
    follow_up = await project_client.agents.create_message_async(
        thread_id=thread.id,
        role="user",
        content="What IT equipment will I be provided with?"
    )
    print(f"\nCreated follow-up message, ID: {follow_up.id}")
    
    # Process the follow-up
    run = await project_client.agents.create_and_process_run_async(
        thread_id=thread.id, 
        assistant_id=agent.id
    )
    print(f"Run finished with status: {run.status}")
    
    # Get updated messages
    messages = await project_client.agents.list_messages_async(thread_id=thread.id)
    
    # Display the assistant's response to the follow-up
    assistant_messages = [m for m in messages if m.role == "assistant"]
    if len(assistant_messages) >= 2:
        print("\nAssistant's follow-up response:")
        print(assistant_messages[-1].content[0].text)
    
    # Clean up
    await project_client.agents.delete_agent_async(agent.id)
    print(f"\nDeleted agent: {agent.id}")
    
    print("\nExample 1 completed!")
    print("\nTry modifying the agent instructions or asking different questions to see how it responds.")

# %% [markdown]
# To run Example 1, uncomment and execute the following cell:

# %%
# await example1_basic_agent()

# %% [markdown]
# ## Example 2: Agent with File Search
#
# In this example, we'll create an agent that can search through company documentation to find relevant information for new employees.

# %%
async def example2_file_search():
    """
    This example demonstrates how to use the file search capability
    to enable an agent to retrieve information from documents.
    """
    print("Running Example 2: Agent with File Search")
    
    # Step 1: Upload file to the agent service
    print("Step 1: Uploading document...")
    file = await project_client.agents.upload_file_and_poll_async(
        file_path="challenge-5/docs/contoso_electronics.pdf", 
        purpose="assistants"
    )
    print(f"Uploaded file, file ID: {file.id}")
    
    # Step 2: Create a vector store with the uploaded file
    print("\nStep 2: Creating vector store...")
    vector_store = await project_client.agents.create_vector_store_and_poll_async(
        file_ids=[file.id], 
        name="employee_handbook"
    )
    print(f"Created vector store, vector store ID: {vector_store.id}")
    
    # Step 3: Create file search tool
    print("\nStep 3: Creating file search tool...")
    file_search = FileSearchTool(vector_store_ids=[vector_store.id])
    
    # Step 4: Create the agent with file search capability
    print("\nStep 4: Creating agent with file search...")
    agent = await project_client.agents.create_agent_async(
        model=model_deployment_name,
        name="Document Search Assistant",
        instructions="""
        You are a Document Search Assistant designed to help employees find information 
        in company documentation. 
        
        When asked a question:
        1. Search the company documents for relevant information
        2. Provide a clear, concise answer based on the search results
        3. Include specific references to the source document when appropriate
        4. If the information isn't in the documents, acknowledge that and provide a general response
        
        Always maintain a helpful, professional tone.
        """,
        tools=file_search.definitions,
        tool_resources=file_search.resources
    )
    
    print(f"Created agent, ID: {agent.id}")
    
    # Step 5: Create a thread and ask questions
    print("\nStep 5: Creating a conversation thread...")
    thread = await project_client.agents.create_thread_async()
    print(f"Created thread, ID: {thread.id}")
    
    # Ask a question that should be answerable from the document
    question = "What is the company's policy on remote work?"
    print(f"\nAsking: '{question}'")
    
    message = await project_client.agents.create_message_async(
        thread_id=thread.id,
        role="user",
        content=question
    )
    
    # Create and process a run
    run = await project_client.agents.create_and_process_run_async(
        thread_id=thread.id, 
        assistant_id=agent.id
    )
    print(f"Run finished with status: {run.status}")
    
    # Get messages in the thread
    messages = await project_client.agents.list_messages_async(thread_id=thread.id)
    
    # Display the assistant's response
    assistant_messages = [m for m in messages if m.role == "assistant"]
    if assistant_messages:
        print("\nAssistant's response:")
        print(assistant_messages[-1].content[0].text)
    
    # Ask a follow-up question
    follow_up = "What equipment is provided to remote workers?"
    print(f"\nAsking follow-up: '{follow_up}'")
    
    await project_client.agents.create_message_async(
        thread_id=thread.id,
        role="user",
        content=follow_up
    )
    
    # Process the follow-up
    run = await project_client.agents.create_and_process_run_async(
        thread_id=thread.id, 
        assistant_id=agent.id
    )
    
    # Get updated messages
    messages = await project_client.agents.list_messages_async(thread_id=thread.id)
    
    # Display the assistant's response to the follow-up
    assistant_messages = [m for m in messages if m.role == "assistant"]
    if len(assistant_messages) >= 2:
        print("\nAssistant's follow-up response:")
        print(assistant_messages[-1].content[0].text)
    
    # Optional: View the run steps to see how the agent used file search
    print("\nViewing run steps to understand file search process...")
    run_steps = await project_client.agents.list_run_steps_async(thread_id=thread.id, run_id=run.id)
    for i, step in enumerate(run_steps):
        print(f"\nStep {i+1}: {step.type}")
        if step.type == "tool_calls" and step.step_details.tool_calls:
            for j, tool_call in enumerate(step.step_details.tool_calls):
                print(f"  Tool call {j+1}: {tool_call.type}")
                if tool_call.type == "file_search":
                    print(f"  Search query: {tool_call.file_search.query}")
    
    # Clean up
    await project_client.agents.delete_agent_async(agent.id)
    print(f"\nDeleted agent: {agent.id}")
    
    print("\nExample 2 completed!")
    print("\nTry modifying this example by:")
    print("1. Uploading different PDF documents")
    print("2. Asking different questions that might be in the documents")
    print("3. Changing the agent instructions to focus on different aspects of the documents")

# %% [markdown]
# To run Example 2, uncomment and execute the following cell:

# %%
# await example2_file_search()

# %% [markdown]
# ## Example 3: Agent with JSON Schema Functions
#
# In this example, we'll create an agent that can provide structured information using JSON schema functions.

# %%
# Define JSON schema for IT support information
class ITSupportInfo(TypedDict):
    issue_type: str
    support_options: List[Dict[str, str]]
    self_help_resources: List[str]
    escalation_path: Dict[str, str]
    average_resolution_time: str

# Define a function to get IT support information
async def get_it_support_information(issue_category: str) -> ITSupportInfo:
    """
    Get information about IT support procedures and resources.
    
    Args:
        issue_category: The category of IT issue (HARDWARE, SOFTWARE, NETWORK, ACCESS, OTHER)
    
    Returns:
        IT support information including support options, self-help resources, and escalation paths
    """
    support_data = {
        "HARDWARE": {
            "issue_type": "Hardware Problems",
            "support_options": [
                {"channel": "IT Help Desk", "contact": "helpdesk@company.com or ext. 5555"},
                {"channel": "Walk-in Support", "contact": "IT Office, Building B, Floor 2"},
                {"channel": "Emergency Support", "contact": "+1 (888) 555-9999 (24/7)"}
            ],
            "self_help_resources": [
                "Hardware Troubleshooting Guide on company intranet",
                "Common Hardware Issues FAQ",
                "Equipment Request Portal for replacements"
            ],
            "escalation_path": {
                "level_1": "IT Help Desk",
                "level_2": "Desktop Support Technician",
                "level_3": "IT Infrastructure Manager"
            },
            "average_resolution_time": "1-2 business days"
        },
        "SOFTWARE": {
            "issue_type": "Software Issues",
            "support_options": [
                {"channel": "IT Help Desk", "contact": "helpdesk@company.com or ext. 5555"},
                {"channel": "Software Support Chat", "contact": "Available on intranet portal"},
                {"channel": "Virtual Support Session", "contact": "Schedule via help desk ticket"}
            ],
            "self_help_resources": [
                "Software Installation Guide",
                "Application Access Request Form",
                "Software Update Schedule",
                "Licensed Software Portal"
            ],
            "escalation_path": {
                "level_1": "IT Help Desk",
                "level_2": "Software Support Specialist",
                "level_3": "Application Support Manager"
            },
            "average_resolution_time": "4-8 business hours"
        },
        "NETWORK": {
            "issue_type": "Network Connectivity",
            "support_options": [
                {"channel": "IT Help Desk", "contact": "helpdesk@company.com or ext. 5555"},
                {"channel": "Network Operations Center", "contact": "noc@company.com"},
                {"channel": "Emergency Support", "contact": "+1 (888) 555-9999 (24/7)"}
            ],
            "self_help_resources": [
                "Wi-Fi Connection Guide",
                "VPN Setup Instructions",
                "Network Status Page",
                "Remote Access Troubleshooting"
            ],
            "escalation_path": {
                "level_1": "IT Help Desk",
                "level_2": "Network Support Technician",
                "level_3": "Network Operations Manager"
            },
            "average_resolution_time": "2-4 business hours"
        },
        "ACCESS": {
            "issue_type": "System Access and Authentication",
            "support_options": [
                {"channel": "IT Help Desk", "contact": "helpdesk@company.com or ext. 5555"},
                {"channel": "Security Office", "contact": "security@company.com"},
                {"channel": "Password Reset Tool", "contact": "Available on login page"}
            ],
            "self_help_resources": [
                "Self-Service Password Reset",
                "Multi-Factor Authentication Setup Guide",
                "System Access Request Form",
                "Security Policies Document"
            ],
            "escalation_path": {
                "level_1": "IT Help Desk",
                "level_2": "Identity Access Management Team",
                "level_3": "Information Security Officer"
            },
            "average_resolution_time": "1-6 business hours"
        },
        "OTHER": {
            "issue_type": "Other IT Issues",
            "support_options": [
                {"channel": "IT Help Desk", "contact": "helpdesk@company.com or ext. 5555"},
                {"channel": "General Support Email", "contact": "support@company.com"}
            ],
            "self_help_resources": [
                "IT Knowledge Base",
                "New Employee IT Orientation Guide",
                "IT Service Catalog"
            ],
            "escalation_path": {
                "level_1": "IT Help Desk",
                "level_2": "IT Support Manager",
                "level_3": "IT Director"
            },
            "average_resolution_time": "1-2 business days"
        }
    }
    
    if issue_category in support_data:
        return support_data[issue_category]
    else:
        return {
            "issue_type": "Unknown Issue Category",
            "support_options": [{"channel": "IT Help Desk", "contact": "helpdesk@company.com or ext. 5555"}],
            "self_help_resources": ["IT Knowledge Base on company intranet"],
            "escalation_path": {"level_1": "IT Help Desk"},
            "average_resolution_time": "Varies by issue type"
        }

# Define JSON schema for office location information
class OfficeLocation(TypedDict):
    name: str
    address: str
    phone: str
    timezone: str
    facilities: List[str]

# Define a function to get office information
async def get_office_information(office_id: str) -> OfficeLocation:
    """
    Get detailed information about a specific company office location.
    
    Args:
        office_id: The identifier for the office location (HQ, NYC, SF, LONDON)
    
    Returns:
        Office location details including address, contact information, and facilities
    """
    office_data = {
        "HQ": {
            "name": "Headquarters",
            "address": "123 Main Street, Seattle, WA 98101",
            "phone": "+1 (206) 555-1234",
            "timezone": "Pacific Time (UTC-8/UTC-7 DST)",
            "facilities": ["Cafeteria", "Gym", "Daycare", "Shuttle Service", "EV Charging"]
        },
        "NYC": {
            "name": "New York Office",
            "address": "555 Broadway, New York, NY 10012",
            "phone": "+1 (212) 555-6789",
            "timezone": "Eastern Time (UTC-5/UTC-4 DST)",
            "facilities": ["Cafeteria", "Gym", "Bike Storage"]
        },
        "SF": {
            "name": "San Francisco Office",
            "address": "101 Market Street, San Francisco, CA 94105",
            "phone": "+1 (415) 555-2345",
            "timezone": "Pacific Time (UTC-8/UTC-7 DST)",
            "facilities": ["Cafeteria", "Game Room", "Rooftop Deck", "EV Charging"]
        },
        "LONDON": {
            "name": "London Office",
            "address": "10 Finsbury Square, London EC2A 1AF, UK",
            "phone": "+44 20 5555 1234",
            "timezone": "Greenwich Mean Time (UTC+0/UTC+1 BST)",
            "facilities": ["Cafeteria", "Bike Storage", "Lounge"]
        }
    }
    
    if office_id in office_data:
        return office_data[office_id]
    else:
        return {
            "name": "Unknown Office",
            "address": "Information not available",
            "phone": "Information not available",
            "timezone": "Information not available",
            "facilities": []
        }

async def example3_json_schema_functions():
    """
    This example demonstrates how to use JSON schema functions
    to provide structured information from an agent.
    """
    print("Running Example 3: Agent with JSON Schema Functions")
    
    # Step 1: Set up the custom functions
    print("Step 1: Setting up custom functions with JSON schema...")
    
    # Define the IT support function
    it_support_function = UserFunction(
        name="get_it_support_information",
        description="Get information about IT support procedures and resources",
        parameters=[
            FunctionParameter(
                name="issue_category",
                description="The category of IT issue (HARDWARE, SOFTWARE, NETWORK, ACCESS, OTHER)",
                type=JsonSchemaType.STRING,
                required=True
            )
        ],
        function_obj=get_it_support_information,
        return_type=JsonObject(
            properties={
                "issue_type": JsonSchemaType.STRING,
                "support_options": JsonSchemaType.ARRAY,
                "self_help_resources": JsonSchemaType.ARRAY,
                "escalation_path": JsonSchemaType.OBJECT,
                "average_resolution_time": JsonSchemaType.STRING
            }
        )
    )
    
    # Define the office location function
    office_info_function = UserFunction(
        name="get_office_information",
        description="Get detailed information about a specific company office location",
        parameters=[
            FunctionParameter(
                name="office_id",
                description="The identifier for the office location (HQ, NYC, SF, LONDON)",
                type=JsonSchemaType.STRING,
                required=True
            )
        ],
        function_obj=get_office_information,
        return_type=JsonObject(
            properties={
                "name": JsonSchemaType.STRING,
                "address": JsonSchemaType.STRING,
                "phone": JsonSchemaType.STRING,
                "timezone": JsonSchemaType.STRING,
                "facilities": JsonSchemaType.ARRAY
            }
        )
    )
    
    # Create function tool
    function_tool = FunctionTool([it_support_function, office_info_function])
    
    # Step 2: Create the agent with these functions
    print("\nStep 2: Creating agent with custom functions...")
    agent = await project_client.agents.create_agent_async(
        model=model_deployment_name,
        name="IT Support and Office Info Assistant",
        instructions="""
        You are an IT Support and Office Information Assistant designed to help employees with IT issues
        and provide information about company office locations.
        
        You can provide:
        - IT support information for different issue categories
        - Office location details including facilities and contact information
        
        Use the functions available to you to retrieve the structured information when appropriate.
        Explain the information in a helpful way to the employee, and format it clearly.
        """,
        tools=function_tool.definitions
    )
    
    print(f"Created agent, ID: {agent.id}")
    
    # Step 3: Interact with the agent to demonstrate the functions
    print("\nStep 3: Creating a conversation thread...")
    thread = await project_client.agents.create_thread_async()
    print(f"Created thread, ID: {thread.id}")
    
    # Ask a question about IT support
    it_question = "I'm having network connectivity issues. What IT support options do I have?"
    print(f"\nAsking IT question: '{it_question}'")
    
    await project_client.agents.create_message_async(
        thread_id=thread.id,
        role="user",
        content=it_question
    )
    
    # Create and process a run
    it_run = await project_client.agents.create_and_process_run_async(
        thread_id=thread.id, 
        assistant_id=agent.id
    )
    print(f"Run finished with status: {it_run.status}")
    
    # Get messages in the thread
    messages = await project_client.agents.list_messages_async(thread_id=thread.id)
    
    # Display the assistant's response
    assistant_messages = [m for m in messages if m.role == "assistant"]
    if assistant_messages:
        print("\nAssistant's response to IT question:")
        print(assistant_messages[-1].content[0].text)
    
    # Ask a question about office location
    office_question = "What facilities are available at the New York office?"
    print(f"\nAsking office question: '{office_question}'")
    
    await project_client.agents.create_message_async(
        thread_id=thread.id,
        role="user",
        content=office_question
    )
    
    # Process the office question
    office_run = await project_client.agents.create_and_process_run_async(
        thread_id=thread.id, 
        assistant_id=agent.id
    )
    
    # Get updated messages
    messages = await project_client.agents.list_messages_async(thread_id=thread.id)
    
    # Display the assistant's response to the office question
    assistant_messages = [m for m in messages if m.role == "assistant"]
    if len(assistant_messages) >= 2:
        print("\nAssistant's response to office question:")
        print(assistant_messages[-1].content[0].text)
    
    # Optional: View the run steps to see how the agent used the functions
    print("\nViewing run steps to understand function use...")
    run_steps = await project_client.agents.list_run_steps_async(thread_id=thread.id, run_id=office_run.id)
    for i, step in enumerate(run_steps):
        print(f"\nStep {i+1}: {step.type}")
        if step.type == "tool_calls" and step.step_details.tool_calls:
            for j, tool_call in enumerate(step.step_details.tool_calls):
                print(f"  Tool call {j+1}: {tool_call.type}")
                if tool_call.type == "function":
                    print(f"  Function name: {tool_call.function.name}")
                    print(f"  Function arguments: {tool_call.function.arguments}")
                    if tool_call.function.output:
                        print(f"  Function output: {tool_call.function.output}")
    
    # Clean up
    await project_client.agents.delete_agent_async(agent.id)
    print(f"\nDeleted agent: {agent.id}")
    
    print("\nExample 3 completed!")
    print("\nTry modifying this example by:")
    print("1. Adding new functions with different data structures")
    print("2. Changing the schema of existing functions")
    print("3. Asking questions that require multiple function calls")

# %% [markdown]
# To run Example 3, uncomment and execute the following cell:

# %%
# await example3_json_schema_functions()

# %% [markdown]
# ## Example 4: Agent with Code Interpreter
#
# In this example, we'll create an agent that can use the code interpreter to solve technical problems.

# %%
async def example4_code_interpreter():
    """
    This example demonstrates how to use the code interpreter capability
    to allow an agent to analyze data and solve technical problems.
    """
    print("Running Example 4: Agent with Code Interpreter")
    
    # Step 1: Create the code interpreter tool
    print("Step 1: Creating code interpreter tool...")
    code_interpreter = CodeInterpreterTool()
    
    # Step 2: Create the agent with code interpreter
    print("\nStep 2: Creating agent with code interpreter...")
    agent = await project_client.agents.create_agent_async(
        model=model_deployment_name,
        name="IT Analysis Assistant",
        instructions="""
        You are an IT Analysis Assistant specialized in helping IT staff analyze 
        logs, debug code issues, and solve technical problems.
        
        When asked technical questions or presented with data:
        1. Use the code interpreter to analyze the data or solve the problem
        2. Write clear, efficient code to address the issue
        3. Explain your analysis and solution in a way that's easy to understand
        4. If appropriate, include visualizations or sample output
        5. Suggest next steps or recommendations based on your findings
        
        Be precise, technical, and focus on practical solutions.
        """,
        tools=code_interpreter.definitions
    )
    
    print(f"Created agent, ID: {agent.id}")
    
    # Step 3: Create a thread and ask a technical question
    print("\nStep 3: Creating a conversation thread...")
    thread = await project_client.agents.create_thread_async()
    print(f"Created thread, ID: {thread.id}")
    
    # Ask a technical question that requires code analysis
    technical_question = """
    I need help analyzing this error log pattern. Can you identify the pattern and explain what might be happening?
    
    ERROR [2023-09-15 08:23:45] Connection timeout: Database connection failed after 30s
    ERROR [2023-09-15 09:45:12] Connection timeout: Database connection failed after 30s
    ERROR [2023-09-15 10:12:33] Connection timeout: Database connection failed after 30s
    ERROR [2023-09-15 11:37:09] Connection timeout: Database connection failed after 30s
    ERROR [2023-09-15 13:05:22] Connection timeout: Database connection failed after 30s
    ERROR [2023-09-16 08:30:45] Connection timeout: Database connection failed after 30s
    """
    
    print("\nAsking technical question about error logs...")
    
    await project_client.agents.create_message_async(
        thread_id=thread.id,
        role="user",
        content=technical_question
    )
    
    # Create and process a run
    run = await project_client.agents.create_and_process_run_async(
        thread_id=thread.id, 
        assistant_id=agent.id
    )
    print(f"Run finished with status: {run.status}")
    
    # Get messages in the thread
    messages = await project_client.agents.list_messages_async(thread_id=thread.id)
    
    # Display the assistant's response
    assistant_messages = [m for m in messages if m.role == "assistant"]
    if assistant_messages:
        print("\nAssistant's analysis:")
        print(assistant_messages[-1].content[0].text)
    
    # Ask another technical question with data
    data_question = """
    I need to analyze our server response times. Here's the data for the last 10 requests (in milliseconds):
    245, 312, 287, 498, 2456, 198, 211, 2389, 259, 301
    
    Can you analyze this data, identify any outliers, and calculate the average response time 
    both with and without outliers?
    """
    
    print("\nAsking question about server response times...")
    
    await project_client.agents.create_message_async(
        thread_id=thread.id,
        role="user",
        content=data_question
    )
    
    # Process the data question
    data_run = await project_client.agents.create_and_process_run_async(
        thread_id=thread.id, 
        assistant_id=agent.id
    )
    
    # Get updated messages
    messages = await project_client.agents.list_messages_async(thread_id=thread.id)
    
    # Display the assistant's response to the data question
    assistant_messages = [m for m in messages if m.role == "assistant"]
    if len(assistant_messages) >= 2:
        print("\nAssistant's data analysis:")
        print(assistant_messages[-1].content[0].text)
    
    # Optional: View the run steps to see the code execution
    print("\nViewing run steps to understand code execution...")
    run_steps = await project_client.agents.list_run_steps_async(thread_id=thread.id, run_id=data_run.id)
    for i, step in enumerate(run_steps):
        print(f"\nStep {i+1}: {step.type}")
        if step.type == "tool_calls" and step.step_details.tool_calls:
            for j, tool_call in enumerate(step.step_details.tool_calls):
                print(f"  Tool call {j+1}: {tool_call.type}")
                if tool_call.type == "code_interpreter":
                    if tool_call.code_interpreter.input:
                        print(f"  Code input: {tool_call.code_interpreter.input}")
                    if tool_call.code_interpreter.outputs:
                        print(f"  Code outputs: {len(tool_call.code_interpreter.outputs)} output(s)")
    
    # Clean up
    await project_client.agents.delete_agent_async(agent.id)
    print(f"\nDeleted agent: {agent.id}")
    
    print("\nExample 4 completed!")
    print("\nTry modifying this example by:")
    print("1. Providing different types of data for analysis")
    print("2. Asking for specific visualizations")
    print("3. Requesting code solutions for specific problems")

# %% [markdown]
# To run Example 4, uncomment and execute the following cell:

# %%
# await example4_code_interpreter()

# %% [markdown]
# ## Example 5: Agent with Multiple Tools
#
# In this final example, we'll create a comprehensive agent that combines multiple tools (file search, JSON schema functions, and code interpreter) to provide a complete employee support experience.

# %%
async def example5_combined_tools():
    """
    This example demonstrates how to combine multiple tools in a single agent
    to create a comprehensive employee support assistant.
    """
    print("Running Example 5: Agent with Multiple Tools")
    
    # Step 1: Set up file search
    print("Step 1: Setting up file search...")
    file = await project_client.agents.upload_file_and_poll_async(
        file_path="challenge-5/docs/contoso_electronics.pdf", 
        purpose="assistants"
    )
    print(f"Uploaded file, file ID: {file.id}")
    
    vector_store = await project_client.agents.create_vector_store_and_poll_async(
        file_ids=[file.id], 
        name="combined_handbook"
    )
    print(f"Created vector store, vector store ID: {vector_store.id}")
    
    file_search = FileSearchTool(vector_store_ids=[vector_store.id])
    
    # Step 2: Set up custom functions
    print("\nStep 2: Setting up custom functions...")
    
    # IT Support function
    it_support_function = UserFunction(
        name="get_it_support_information",
        description="Get information about IT support procedures and resources",
        parameters=[
            FunctionParameter(
                name="issue_category",
                description="The category of IT issue (HARDWARE, SOFTWARE, NETWORK, ACCESS, OTHER)",
                type=JsonSchemaType.STRING,
                required=True
            )
        ],
        function_obj=get_it_support_information,
        return_type=JsonObject(
            properties={
                "issue_type": JsonSchemaType.STRING,
                "support_options": JsonSchemaType.ARRAY,
                "self_help_resources": JsonSchemaType.ARRAY,
                "escalation_path": JsonSchemaType.OBJECT,
                "average_resolution_time": JsonSchemaType.STRING
            }
        )
    )
    
    # Office location function
    office_info_function = UserFunction(
        name="get_office_information",
        description="Get detailed information about a specific company office location",
        parameters=[
            FunctionParameter(
                name="office_id",
                description="The identifier for the office location (HQ, NYC, SF, LONDON)",
                type=JsonSchemaType.STRING,
                required=True
            )
        ],
        function_obj=get_office_information,
        return_type=JsonObject(
            properties={
                "name": JsonSchemaType.STRING,
                "address": JsonSchemaType.STRING,
                "phone": JsonSchemaType.STRING,
                "timezone": JsonSchemaType.STRING,
                "facilities": JsonSchemaType.ARRAY
            }
        )
    )
    
    # Create function tool
    function_tool = FunctionTool([it_support_function, office_info_function])
    
    # Step 3: Set up code interpreter
    print("\nStep 3: Setting up code interpreter...")
    code_interpreter = CodeInterpreterTool()
    
    # Step 4: Create a toolset with all tools
    print("\nStep 4: Creating toolset with all tools...")
    toolset = ToolSet()
    toolset.add(function_tool)
    toolset.add(code_interpreter)
    
    # Step 5: Create the agent with all tools
    print("\nStep 5: Creating comprehensive agent with all tools...")
    agent = await project_client.agents.create_agent_async(
        model=model_deployment_name,
        name="Comprehensive Employee Support Assistant",
        instructions="""
        You are a Comprehensive Employee Support Assistant that combines document search, 
        structured information, and technical analysis capabilities.
        
        You can:
        1. Search company documentation to answer policy questions
        2. Provide structured information about office locations and IT support
        3. Analyze technical problems using code
        
        Use the most appropriate tool for each question:
        - For policy questions, search the company documents
        - For office or IT support questions, use the relevant functions
        - For technical issues, use the code interpreter
        
        Always be helpful, accurate, and concise in your responses.
        """,
        toolset=toolset,
        file_search_tool=file_search
    )
    
    print(f"Created agent, ID: {agent.id}")
    
    # Step 6: Create a thread and demonstrate each capability
    print("\nStep 6: Creating a conversation thread...")
    thread = await project_client.agents.create_thread_async()
    print(f"Created thread, ID: {thread.id}")
    
    # Example questions to demonstrate different tools
    questions = [
        {
            "type": "document search",
            "question": "What is the company's policy on vacation time?"
        },
        {
            "type": "function call",
            "question": "What facilities are available at the SF office?"
        },
        {
            "type": "code interpreter",
            "question": """
            I'm trying to understand our system performance. Here's some CPU utilization data:
            32%, 45%, 67%, 88%, 91%, 85%, 73%, 62%, 51%, 43%, 38%
            
            Can you analyze this trend and create a visualization?
            """
        }
    ]
    
    # Process each question
    for i, q in enumerate(questions):
        print(f"\nQuestion {i+1} ({q['type']}): {q['question']}")
        
        await project_client.agents.create_message_async(
            thread_id=thread.id,
            role="user",
            content=q['question']
        )
        
        run = await project_client.agents.create_and_process_run_async(
            thread_id=thread.id, 
            assistant_id=agent.id
        )
        print(f"Run finished with status: {run.status}")
        
        # Get messages
        messages = await project_client.agents.list_messages_async(thread_id=thread.id)
        
        # Display the assistant's response
        assistant_messages = [m for m in messages if m.role == "assistant"]
        if len(assistant_messages) >= i+1:
            print(f"\nAssistant's response to {q['type']} question:")
            print(assistant_messages[-1].content[0].text)
    
    # Clean up
    await project_client.agents.delete_agent_async(agent.id)
    print(f"\nDeleted agent: {agent.id}")
    
    print("\nExample 5 completed!")
    print("\nThis example demonstrates how to combine multiple tools in a single agent.")
    print("Try experimenting with different combinations of tools and instructions to see how they work together.")

# %% [markdown]
# To run Example 5, uncomment and execute the following cell:

# %%
# await example5_combined_tools()

# %% [markdown]
# ## Run All Examples
#
# To run all examples in sequence, uncomment and execute the following cell:

# %%
# Run multiple examples in sequence
async def run_all_examples():
    await example1_basic_agent()
    print("\n" + "="*80 + "\n")
    
    await example2_file_search()
    print("\n" + "="*80 + "\n")
    
    await example3_json_schema_functions()
    print("\n" + "="*80 + "\n")
    
    await example4_code_interpreter()
    print("\n" + "="*80 + "\n")
    
    await example5_combined_tools()

# %%
# await run_all_examples()

# %% [markdown]
# ## Documents for File Search
#
# For the file search examples (Examples 2 and 5), we use the PDF document located at `challenge-5/docs/contoso_electronics.pdf`. However, you can experiment with your own PDF documents to enhance the agent's knowledge.
#
# To use your own documents:
#
# 1. Replace `"challenge-5/docs/contoso_electronics.pdf"` in the code with the path to your document
# 2. The document will be processed and indexed for the agent to search
# 3. You can ask questions related to the content of your document
#
# Documents that work well include:
# - Employee handbooks
# - Company policies
# - IT procedure guides
# - Onboarding materials
# - Product documentation
#
# ## Conclusion
#
# In this notebook, we explored the Azure AI Agents SDK through focused, independent examples that demonstrate its key capabilities:
#
# 1. Creating basic agents for conversational interactions
# 2. Enabling document search through file search tools
# 3. Returning structured data with JSON schema functions
# 4. Solving technical problems with code interpreter
# 5. Combining multiple tools for comprehensive capabilities
#
# Each example is designed to be independently runnable, allowing you to experiment with specific features that interest you. Feel free to modify the examples, combine different techniques, or adapt them to your specific use cases.
#
# The Azure AI Agents SDK provides a powerful platform for building intelligent agents that can access various tools and resources to better assist users with their tasks. 