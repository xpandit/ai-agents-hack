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
# # Challenge 4: Functions & Plugins with Semantic Kernel
#
# In this challenge, we'll explore Microsoft's Semantic Kernel framework and learn how to create powerful AI agents by combining prompts with custom functions.
#
# ## What is Semantic Kernel?
#
# Semantic Kernel (SK) is an open-source SDK that integrates Large Language Models (LLMs) with programming languages. It provides a structured way to combine AI capabilities with custom code, external data sources, and API services.
#
# What makes Semantic Kernel powerful for building AI agents:
#
# - **Plugin Architecture**: Create reusable components that extend AI capabilities
# - **Function Calling**: Allow LLMs to invoke your custom code when needed
# - **Seamless Integration**: Combine AI prompts and traditional programming in a unified workflow
# - **Memory & Context Management**: Maintain state and manage conversation history
# - **Enterprise Readiness**: Designed for production applications with scalability in mind
#
# ## Key Concepts in Semantic Kernel
#
# ### 1. Kernel
#
# The kernel is the central orchestrator in Semantic Kernel. It:
# - Manages LLM connections and contexts
# - Handles function registration and execution
# - Coordinates plugins and their interactions
#
# ### 2. Functions
#
# Semantic Kernel supports two types of functions:
#
# - **Native Functions**: Traditional code (Python, C#, etc.) that performs specific tasks
# - **Semantic Functions**: AI prompt templates that guide the LLM to perform tasks
#
# ### 3. Plugins
#
# Plugins are collections of related functions (both native and semantic) that work together to provide specific capabilities.

# %% [markdown]
# ## Setting Up Our Environment
#
# First, let's install the necessary packages and set up our Semantic Kernel environment.

# %%
# !pip install semantic-kernel openai python-dotenv

# %%
import os
import json
import time
import zipfile  # For zip file operations
import random   # For random password generation
import string   # For password character sets
import shutil   # For file operations
import re       # For regex operations
from typing import List, Dict, Any, Annotated, Optional
from IPython.display import display, HTML, Markdown
from pathlib import Path  # For path manipulations

import semantic_kernel as sk
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.functions.kernel_function_metadata import KernelFunctionMetadata
from semantic_kernel.contents import FunctionCallContent, FunctionResultContent  # For function call tracking
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIPromptExecutionSettings,
)

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Azure OpenAI credentials
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Create a kernel instance
kernel = sk.Kernel()

# Add Azure OpenAI chat service
kernel.add_service(
    AzureChatCompletion(
        service_id="azure_chat_completion",
        deployment_name=azure_deployment,
        endpoint=azure_openai_endpoint,
        api_key=azure_openai_key,
        api_version=azure_api_version
    )
)

print("Semantic Kernel initialized successfully!")

# %% [markdown]
# ## 1. Creating Semantic Functions (Prompts as Functions)
#
# Semantic functions are AI prompts wrapped as callable functions. They help standardize how you interact with LLMs and make prompts reusable.
#
# Let's create a simple semantic function that helps with employee onboarding by explaining company acronyms:

# %%
# Define a simple semantic function as a string
acronym_explainer_prompt = """
You are an expert on company terminology and acronyms.
Given a company acronym, explain what it stands for and provide a brief description.

Acronym: {{$acronym}}

Explanation:
"""

# Add the semantic function to the plugin
acronym_explainer = kernel.add_function(
    plugin_name="acronym",
    function_name="explain",
    prompt=acronym_explainer_prompt,
    description="Explains company acronyms"
)

# %% [markdown]
# Now let's test our semantic function by asking it to explain some common IT acronyms:

# %%
# Invoke the semantic function
result = await kernel.invoke(
    plugin_name="acronym",
    function_name="explain",
    arguments=KernelArguments(acronym="VPN")
)

print(result)

# %%
# Try with another acronym
result = await kernel.invoke(
    plugin_name="acronym",
    function_name="explain",
    arguments=KernelArguments(acronym="SSO")
)

print(result)

# %% [markdown]
# ## 2. Creating Native Functions (Code as Functions)
#
# Native functions are regular code functions that you register with the kernel. They can be called by the AI or directly from your code. Let's create a simple IT support plugin with native functions for common tasks.

# %%
class ITSupportPlugin:
    """Plugin for common IT support tasks."""
    
    @kernel_function(description="Get setup instructions for various company tools")
    def get_setup_instructions(self, tool: str) -> str:
        """
        Get step-by-step setup instructions for company tools like email, VPN, etc.
        """
        tools = {
            "email": "1. Open Outlook\n2. Click on 'Add Account'\n3. Enter your company email\n4. Enter the temporary password from your welcome email\n5. Follow the prompts to set up multi-factor authentication",
            
            "vpn": "1. Download the company VPN client from the IT portal\n2. Install the client\n3. Launch the client\n4. Enter your company email and password\n5. Select the appropriate server location\n6. Click 'Connect'",
            
            "slack": "1. Download Slack from slack.com or your app store\n2. Open Slack\n3. Find our workspace by entering company.slack.com\n4. Sign in with your company email\n5. Join the #welcome and #general channels",
            
            "github": "1. Create a GitHub account if you don't have one\n2. Send your GitHub username to IT\n3. Accept the invitation to join the company organization\n4. Set up two-factor authentication\n5. Install Git on your computer"
        }
        
        return tools.get(tool.lower(), "Instructions not found for the specified tool. Please contact IT helpdesk.")
    
    @kernel_function(description="Check system requirements for company software")
    def check_system_requirements(self, software: str) -> str:
        """
        Get minimum system requirements for running company software.
        """
        requirements = {
            "design_suite": "Windows 10/11 or macOS 10.15+\nProcessor: Intel i5/AMD Ryzen 5 or better\nRAM: 16GB minimum\nStorage: 256GB SSD\nGraphics: Dedicated GPU with 4GB VRAM",
            
            "development_environment": "Windows 10/11, macOS 10.15+, or Linux\nProcessor: Intel i7/AMD Ryzen 7 or better\nRAM: 32GB recommended\nStorage: 512GB SSD\nAdditional: Docker compatibility required",
            
            "office_suite": "Windows 10/11 or macOS 10.14+\nProcessor: 1.6 GHz dual-core\nRAM: 8GB minimum\nStorage: 10GB available space\nDisplay: 1280x800 resolution",
            
            "video_conferencing": "Windows 10/11 or macOS 10.13+\nProcessor: Dual-core 2GHz+\nRAM: 4GB minimum\nNetwork: Broadband connection\nCamera: HD webcam\nAudio: Microphone and speakers"
        }
        
        return requirements.get(software.lower(), "System requirements not found for the specified software. Please contact IT helpdesk.")

# Register the plugin with the kernel
it_support = kernel.add_plugin(ITSupportPlugin(), "it_support")

# Create a setup guide formatter prompt
setup_guide_prompt = """
You are a technical writer specializing in creating clear, well-formatted setup guides for developers.
Your task is to create a comprehensive developer setup guide that combines system requirements and 
tool setup instructions into a cohesive, easy-to-follow document.

## System Requirements
{{$requirements}}

## Tools to Set Up
{{$tool_instructions}}

Please format this information into a professional, well-organized setup guide that:
1. Has a clear introduction explaining the purpose of the guide
2. Organizes the system requirements in a readable format
3. Presents the tool setup instructions in a logical sequence with clear headings
4. Adds helpful tips and best practices where appropriate
5. Includes a troubleshooting section for common issues
6. Ends with next steps and who to contact for help

Format the guide to be welcoming to new employees and easy to follow.
"""

# Create a setup guide plugin with the semantic function
setup_guide_formatter = kernel.add_function(
    function_name="create_developer_guide",
    plugin_name="setup_guide",
    prompt=setup_guide_prompt,
    description="Creates a well-formatted developer setup guide"
)

# %% [markdown]
# Let's test our native functions:

# %%
# Get setup instructions for VPN
result = await kernel.invoke(
    it_support["get_setup_instructions"],
    KernelArguments(tool="vpn")
)

print(result)

# %%
# Check system requirements for the development environment
result = await kernel.invoke(
    it_support["check_system_requirements"],
    KernelArguments(software="development_environment")
)

print(result)

# %% [markdown]
# ### Manually Calling Functions and Combining Results
#
# Now let's see how we can manually call native functions and combine their results in a more complex workflow. This demonstrates how you can orchestrate function calls in your application code:

# %%
# Manually get results from multiple functions
vpn_instructions = await kernel.invoke(
    it_support["get_setup_instructions"],
    KernelArguments(tool="vpn")
)

github_instructions = await kernel.invoke(
    it_support["get_setup_instructions"],
    KernelArguments(tool="github")
)

slack_instructions = await kernel.invoke(
    it_support["get_setup_instructions"],
    KernelArguments(tool="slack")
)

dev_environment_requirements = await kernel.invoke(
    it_support["check_system_requirements"],
    KernelArguments(software="development_environment")
)

# Now combine the results using our semantic function
all_tool_instructions = f"""
### VPN Setup
{vpn_instructions}

### GitHub Setup
{github_instructions}

### Slack Setup
{slack_instructions}
"""

# Use the prompt function to format everything nicely
formatted_guide = await kernel.invoke(
    function_name="create_developer_guide",
    plugin_name="setup_guide",
    arguments=KernelArguments(
        requirements=dev_environment_requirements,
        tool_instructions=all_tool_instructions
    )
)

print(formatted_guide)

# %% [markdown]
# This example demonstrates how Semantic Kernel allows you to:
#
# 1. Call native functions directly from your code to get raw data and information
# 2. Combine the results from multiple function calls
# 3. Use a semantic function (prompt) to format and enhance the results with AI-powered natural language
#
# This pattern is very powerful - we use code for precise data retrieval and computation, then use AI for formatting,
# explanation, and presentation. This gives us the best of both worlds: the reliability of code with the
# natural language capabilities of AI.

# %% [markdown]
# ## 3. Creating a Semantic Function that Calls Native Functions
#
# Now let's create a semantic function that can call our native functions directly from the prompt template. This demonstrates how AI can use functions as tools within its reasoning process:

# %%
# Define a prompt that correctly calls native functions using proper SK syntax
developer_guide_prompt = """
You are an IT onboarding specialist creating a complete setup guide for new developers.

# Developer Onboarding Guide

## System Requirements
Here are the system requirements for your development environment:

{{it_support.check_system_requirements "development_environment"}}

## Required Tools Setup

### Email Setup
{{it_support.get_setup_instructions "email"}}

### VPN Setup
{{it_support.get_setup_instructions "vpn"}}

### GitHub Setup
{{it_support.get_setup_instructions "github"}}

### Slack Setup
{{it_support.get_setup_instructions "slack"}}

## Questions?
If you have any questions about this guide or need additional assistance, please contact the IT helpdesk at helpdesk@company.com or extension 1234.

Welcome to the team!
"""

# Add the semantic function to the kernel
dev_guide_generator = kernel.add_function(
    plugin_name="dev_onboarding",
    function_name="create_guide",
    prompt=developer_guide_prompt,
    description="Creates a comprehensive developer onboarding guide with system requirements, setup instructions, and first-week checklist"
)

# %% [markdown]
# Let's test our developer guide semantic function that calls multiple native functions directly from the prompt template.
#
# Note the proper Semantic Kernel syntax for function calls in prompt templates:
# - `{{plugin_name.function_name}}` - Calls a function with no parameters
# - `{{plugin_name.function_name "literal_value"}}` - Calls a function with a literal string value
# - `{{plugin_name.function_name $variable_name}}` - Calls a function with a variable defined in the KernelArguments
# - Function calls can NOT include a $ prefix for parameters directly inside the function call

# %%
# Invoke the semantic function with access to all required plugins
dev_guide = await kernel.invoke(
    plugin_name="dev_onboarding",
    function_name="create_guide",
    arguments=KernelArguments(
        plugins=[it_support]  # Provide access to all plugins
    )
)

print(dev_guide)

# %% [markdown]
# This example demonstrates the power of Semantic Kernel's prompt templating with function calling. Notice how:
#
# 1. **Proper Function Call Syntax**: Following the SK documentation, we use:
#    - `{{plugin_name.function_name}}` for no parameters
#    - `{{plugin_name.function_name "literal"}}` for string literals
#    - `{{plugin_name.function_name $variable}}` for variables (note the space between function name and parameter)
#
# 2. **Nested Function Calls**: Functions can be nested, like `calculator.add $value1 (calculator.multiply $value2 $value3)`
#
# 3. **Multiple Plugins**: We can incorporate functions from different plugins in the same template
#
# This hardcoded function calling approach is powerful for scenarios where you want deterministic behavior
# rather than letting the AI decide when to call functions. It ensures that specific calculations or data retrievals
# are always performed, which is perfect for applications like budget calculators or onboarding guides
# where the structure is consistent.

# %% [markdown]
# ## 4. Function Calling: Manual vs. Automatic
#
# Semantic Kernel supports two modes of function calling:
#
# 1. **Manual Function Calling**: Explicitly calling functions from your code
# 2. **Automatic Function Calling**: Letting the AI decide which functions to call based on the context
#
# We've already seen manual function calling. Now let's set up automatic function calling:

# %%
# First, let's create a calculator plugin for our assistant to use
class CalculatorPlugin:
    """Plugin for performing mathematical calculations."""
    
    @kernel_function(description="Add two numbers together")
    def add(self, number1: Annotated[float, "The first number"], number2: Annotated[float, "The second number"]) -> str:
        """Add two numbers and return the result."""
        print(f"Adding {number1} and {number2}")
        return str(float(number1) + float(number2))
    
    @kernel_function(description="Subtract the second number from the first number")
    def subtract(self, number1: Annotated[float, "The first number"], number2: Annotated[float, "The second number"]) -> str:
        """Subtract number2 from number1 and return the result."""
        return str(float(number1) - float(number2))
    
    @kernel_function(description="Multiply two numbers together")
    def multiply(self, number1: Annotated[float, "The first number"], number2: Annotated[float, "The second number"]) -> str:
        """Multiply two numbers and return the result."""
        return str(float(number1) * float(number2))
    
    @kernel_function(description="Divide the first number by the second number")
    def divide(self, number1: Annotated[float, "The first number"], number2: Annotated[float, "The second number"]) -> str:
        """Divide number1 by number2 and return the result."""
        if float(number2) == 0:
            return "Error: Cannot divide by zero"
        return str(float(number1) / float(number2))
    
    @kernel_function(description="Calculate the total cost for multiple items")
    def calculate_total_cost(
        self, 
        item_cost: Annotated[float, "The cost per item"],
        quantity: Annotated[int, "The number of items"]
    ) -> str:
        """Calculate the total cost for a quantity of items at a given cost per item."""
        return str(float(item_cost) * int(quantity))

# Register the calculator plugin with the kernel
calculator = kernel.add_plugin(CalculatorPlugin(), "calculator")

# Create an HR assistant with access to IT support and calculator plugins
hr_assistant_prompt = """
You are an HR assistant helping new employees get set up with their equipment and software.
You have access to IT support information and can perform calculations to help with budgeting.

Use the available functions to provide the most helpful response possible.

User: {{$input}}
Assistant:
"""

# Create the HR assistant
hr_assistant = kernel.add_function(
    function_name="respond",
    plugin_name="hr_assistant",
    prompt=hr_assistant_prompt,
    description="Responds to employee onboarding questions",
)

# %% [markdown]
# Now let's create a chat interface with our HR assistant:

# %%
async def chat_with_hr_assistant(question: str):
    # Get plugins we want the assistant to use
    available_plugins = [
        calculator,
        it_support
    ]

    # Let the assistant decide which functions to call
    result = await kernel.invoke(
        function_name="respond",
        plugin_name="hr_assistant",
        arguments=KernelArguments(
            settings=OpenAIPromptExecutionSettings(
                function_choice_behavior=FunctionChoiceBehavior.Auto()
            ),
            input=question
        ),
    )

    # Print function call details if present
    function_calls_found = False
    print("\n--- Response from HR Assistant ---")
    print(result)
    print("\n--- Function Calls ---")
    
    # Check for function calls in the messages
    for message in result.metadata.get('messages', []):
        if hasattr(message, 'items'):
            for item in message.items:
                if isinstance(item, FunctionCallContent):
                    function_calls_found = True
                    print(f"Function called: {item.plugin_name}.{item.function_name}")
                    print(f"Arguments: {item.arguments}")
                elif isinstance(item, FunctionResultContent):
                    print(f"Function result: {item.result}")
    
    if not function_calls_found:
        print("No function calls detected.")

    return result

# %% [markdown]
# Let's test our HR assistant with some questions:

# %%
# First interaction - setting up for a new developer
question = "what is 2+3"
response = await chat_with_hr_assistant(question)
print(response)

# %%
# Second interaction - budget question
question = "Our team is onboarding 5 new designers. Each needs a high-end laptop ($2000), two monitors ($400 each), and a design tablet ($800). What's our total equipment budget for the team? Can you break down the cost per item type?"
response = await chat_with_hr_assistant(question)
print(response)

# %%
# Third interaction - budget calculation question
question = "We need to order equipment for 3 new developers and 2 designers. Developers need laptops ($1800 each), monitors ($350 each), and mechanical keyboards ($150 each). Designers need laptops ($2200 each), monitors ($400 each), and design tablets ($600 each). Please calculate the total budget needed and break it down by role and item type."
response = await chat_with_hr_assistant(question)
print(response)

# %% [markdown]
# ## Conclusion
#
# In this challenge, we've explored the key features of Semantic Kernel that make it powerful for building AI agents:
#
# 1. **Semantic Functions**: Creating reusable AI prompts as functions
# 2. **Native Functions**: Integrating code with AI capabilities
# 3. **Plugins**: Organizing related functions into logical groups
# 4. **Function Calling**: Giving the AI the ability to call your functions when needed
# 5. **Chat Context**: Maintaining conversation state across interactions
#
# Our onboarding assistant demonstrates how Semantic Kernel can be used to build practical applications that combine AI with custom business logic. By structuring your application this way, you get:
#
# - **Modularity**: Easy to extend with new functions or plugins
# - **Reusability**: Components can be shared across different AI agents
# - **Flexibility**: Clear separation between AI reasoning and business logic
# - **Maintainability**: Changes to functions don't require retraining AI models
#
# In the next challenge, we'll explore tool usage and agentic RAG, taking our AI assistant capabilities even further! 

# %% [markdown]
# ## 5. File System Password Management
#
# Let's extend our capabilities by creating a password management assistant that can:
#
# 1. Read and update passwords stored in a JSON file
# 2. Create and modify password-protected zip files
# 3. Generate secure passwords
#
# This demonstrates how Semantic Kernel can interact with the filesystem to perform security-related operations.

# %%
import json
import os
import zipfile
import random
import string
import shutil
import re
from pathlib import Path

class PasswordManagerPlugin:
    """Plugin for password management operations."""
    
    def __init__(self, base_dir="docs/security"):
        self.base_dir = base_dir
        # Ensure full path is used
        if not os.path.isabs(base_dir):
            self.base_dir = os.path.join(os.getcwd(), "challenge-4", base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        self.password_file = os.path.join(self.base_dir, "passwords.json")
    
    @kernel_function(description="Reads the current password for a specific system")
    def read_password(self, 
                     system_name: Annotated[str, "The name of the system to get password for"]
                    ) -> str:
        """Read the current password for a specific system from the password file."""
        try:
            if not os.path.exists(self.password_file):
                return "Password file not found."
            
            with open(self.password_file, 'r') as f:
                passwords = json.load(f)
            
            if 'systems' not in passwords or system_name not in passwords['systems']:
                return f"No password found for system: {system_name}"
            
            return f"Current password for {system_name}: {passwords['systems'][system_name]}"
        
        except Exception as e:
            return f"Error reading password: {str(e)}"
    
    @kernel_function(description="Updates the password for a specific system")
    def update_password(self, 
                       system_name: Annotated[str, "The name of the system to update password for"],
                       new_password: Annotated[str, "The new password to set"]
                      ) -> str:
        """Update the password for a specific system in the password file."""
        try:
            passwords = {}
            if os.path.exists(self.password_file):
                with open(self.password_file, 'r') as f:
                    passwords = json.load(f)
            
            if 'systems' not in passwords:
                passwords['systems'] = {}
            
            # Store old password for confirmation message
            old_password = passwords['systems'].get(system_name, "None")
            
            # Update the password
            passwords['systems'][system_name] = new_password
            
            with open(self.password_file, 'w') as f:
                json.dump(passwords, f, indent=2)
            
            return f"Password for {system_name} updated successfully. Changed from '{old_password}' to '{new_password}'."
        
        except Exception as e:
            return f"Error updating password: {str(e)}"
    
    @kernel_function(description="Lists all systems with stored passwords")
    def list_systems(self) -> str:
        """List all systems that have passwords stored in the password file."""
        try:
            if not os.path.exists(self.password_file):
                return "Password file not found."
            
            with open(self.password_file, 'r') as f:
                passwords = json.load(f)
            
            if 'systems' not in passwords or not passwords['systems']:
                return "No systems found with stored passwords."
            
            systems = list(passwords['systems'].keys())
            return "Systems with stored passwords:\n- " + "\n- ".join(systems)
        
        except Exception as e:
            return f"Error listing systems: {str(e)}"
    
    @kernel_function(description="Updates the password for a zip file")
    def update_zip_password(self, 
                           zip_name: Annotated[str, "Name of the zip file without extension"],
                           new_password: Annotated[str, "The new password to set for the zip file"],
                           old_password: Annotated[str, "The current password of the zip file"]
                          ) -> str:
        """Update the password for a zip file by creating a new zip with the new password."""
        try:
            zip_path = os.path.join(self.base_dir, f"{zip_name}.zip")
            if not os.path.exists(zip_path):
                return f"Zip file not found: {zip_name}.zip"
            
            # Create temporary directory
            temp_dir = os.path.join(self.base_dir, "temp_extract")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Extract the zip with old password
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(path=temp_dir, pwd=old_password.encode())
            except Exception as e:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return f"Failed to extract zip file. Check if the old password is correct: {str(e)}"
            
            # Create a new zip with new password
            new_zip_path = os.path.join(self.base_dir, f"{zip_name}_new.zip")
            with zipfile.ZipFile(new_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zf.write(file_path, arcname, zipfile.ZIP_DEFLATED)
            
            # Use command line zip to set password (Python's zipfile doesn't support encryption directly)
            os.system(f'cd "{self.base_dir}" && zip -P "{new_password}" "{zip_name}_new.zip" -r .')
            
            # Replace the old zip with the new one
            os.remove(zip_path)
            os.rename(new_zip_path, zip_path)
            
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return f"Password for {zip_name}.zip updated successfully."
        
        except Exception as e:
            return f"Error updating zip password: {str(e)}"
    
    @kernel_function(description="Generates a strong random password")
    def generate_password(self, 
                         length: Annotated[int, "Length of the password to generate"] = 16,
                         include_special_chars: Annotated[bool, "Whether to include special characters"] = True
                        ) -> str:
        """Generate a strong random password of specified length."""
        try:
            if length < 8:
                return "Password length should be at least 8 characters."
            
            chars = string.ascii_letters + string.digits
            if include_special_chars:
                chars += string.punctuation
            
            # Generate a password with at least one of each required character type
            password = [
                random.choice(string.ascii_lowercase),
                random.choice(string.ascii_uppercase),
                random.choice(string.digits)
            ]
            
            if include_special_chars:
                password.append(random.choice(string.punctuation))
            
            # Fill the rest with random characters
            password.extend(random.choice(chars) for _ in range(length - len(password)))
            
            # Shuffle the password characters
            random.shuffle(password)
            
            return ''.join(password)
        
        except Exception as e:
            return f"Error generating password: {str(e)}"

# Register the password manager plugin with the kernel
password_manager = kernel.add_plugin(PasswordManagerPlugin(), "password_manager")

# Create a semantic function for the password change assistant
password_assistant_prompt = """
You are a helpful security assistant specializing in password management.
You can help users change passwords for various systems and manage password-protected files.

Use the available password management functions to assist the user with their password-related tasks.
Always confirm the changes made and provide clear, security-conscious advice.

For password changes, suggest strong passwords unless the user specifies otherwise.
For security purposes, avoid suggesting common or easily guessable passwords.

User: {{$input}}
Assistant:
"""

# Add the semantic function to the kernel
password_assistant = kernel.add_function(
    plugin_name="security_assistant",
    function_name="respond",
    prompt=password_assistant_prompt,
    description="Responds to password management and security questions"
)

# %% [markdown]
# Let's set up a chat interface with our password management assistant:

# %%
async def chat_with_password_assistant(question: str):
    # Configure execution settings to enable function calling
    execution_settings = OpenAIPromptExecutionSettings(
        function_choice_behavior=FunctionChoiceBehavior.Auto()
    )
    
    # Invoke the assistant with the user's question and function calling enabled
    result = await kernel.invoke(
        plugin_name="security_assistant",
        function_name="respond",
        arguments=KernelArguments(
            settings=execution_settings,
            input=question
        )
    )
    
    print("\n--- Response from Password Assistant ---")
    print(result)
    
    print("\n--- Function Calls ---")
    for message in result.metadata.get('messages', []):
        if hasattr(message, 'items'):
            for item in message.items:
                if isinstance(item, FunctionCallContent):
                    print(f"Function called: {item.plugin_name}.{item.function_name}")
                elif isinstance(item, FunctionResultContent):
                    print(f"Function result: {item.result}")
    
    return result

# %% [markdown]
# Now, let's test our password management assistant with some scenarios:

# %%
# First, let's check what systems have passwords stored
question = "What systems do we have passwords stored for?"
response = await chat_with_password_assistant(question)

# %%
# Let's update a password
question = "I need to change the VPN password to a more secure one. Can you help me update it?"
response = await chat_with_password_assistant(question)

# %%
# Let's check the current password for a system
question = "What's the current password for the GitHub system?"
response = await chat_with_password_assistant(question)

# %%
# Let's update a zip file password
question = "I need to change the password for the confidential.zip file. The current password is 'oldZipPass123'. Can you generate a secure password and update it?"
response = await chat_with_password_assistant(question)

# %% [markdown]
# ## Summary
#
# In this extended challenge, we've added a practical application of Semantic Kernel by creating a password management assistant. This showcases how:
#
# 1. **Native Functions can interact with the filesystem** - Reading and writing to password files
# 2. **Security operations can be encapsulated** - Password generation and file encryption
# 3. **AI assistants can manipulate sensitive data safely** - By using functions as a controlled interface
#
# This pattern is particularly powerful for enterprise applications where security is paramount. By creating a controlled interface through native functions, we ensure that:
#
# - The AI can't directly access or manipulate sensitive files without going through validated code
# - Security best practices are enforced programmatically
# - Complex operations (like zip password changing) are abstracted into simple interfaces
#
# This example provides a foundation that could be extended to more comprehensive security operations, such as certificate management, encryption key rotation, or secure backup systems. 
