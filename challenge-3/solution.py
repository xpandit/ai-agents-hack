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
# # Challenge 3: Fundamentals of Prompt Engineering & Evaluation
#
# In this challenge, we'll explore core principles of effective prompt engineering, focusing on practical techniques for crafting prompts and evaluating them using Microsoft's tools. We'll apply these concepts to create effective prompts for HR policy explanation during employee onboarding - a common use case for AI assistants in the workplace.
#
# <!-- This is a demo comment added to show Jupytext pairing in action -->
#
# ## 1. Core Principles of Effective Prompt Crafting
#
# Effective prompts follow these key principles:
#
# 1. **Role & Context Setting**: Define the AI's role and provide context for the task. For example, instructing the AI to act as "an HR specialist helping new employees understand company policies."
#
# 2. **Clear Instructions**: Provide specific, unambiguous directions about what you want the AI to do. For instance, "Explain the company's leave policy in simple language that new employees can easily understand."
#
# 3. **Input/Output Format Specification**: Clearly define how to format inputs and expected outputs. For example, "Present the information in a Q&A format with common questions new employees might have."
#
# 4. **Few-Shot Examples**: Provide examples demonstrating desired input/output patterns to guide the AI's responses. For instance, giving an example of how a policy explanation should be structured.
#
# 5. **Chain-of-Thought Prompting**: Guide the model through a reasoning process. For example, "First explain the basic entitlement, then outline the procedure for requesting leave, and finally describe any special conditions."
#
# 6. **System vs User Prompts**: Utilize system prompts for personality and user prompts for specific requests. For instance, setting the system prompt to establish the AI as an HR assistant, and using user prompts for specific policy questions.
#
# When creating prompts for HR policy explanation during employee onboarding, these principles help ensure that explanations are clear, consistent, and tailored to new employees' needs.

# %% [markdown]
# ## 2. Setting up Our Environment
#
# First, let's install the necessary packages for our prompt engineering and evaluation work. We'll use the Azure OpenAI client for generating responses and the Azure AI Evaluation SDK for measuring prompt effectiveness.

# %%
# !pip install openai python-dotenv numpy pandas matplotlib azure-ai-evaluation ipywidgets

# %%
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI


# Load environment variables
load_dotenv()

# Set up AsyncOpenAI client with Azure credentials
client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
    api_key=os.getenv('AZURE_OPENAI_KEY'),
    api_version = "2024-12-01-preview"
)

# %% [markdown]
# ## 3. Interactive Prompt Development
#
# Let's set up a function to test our prompts interactively. This function will allow us to send prompts to the AI model and receive responses, which is essential for the iterative process of prompt refinement. 
#
# Interactive prompt development is crucial for HR policy explanation as it allows us to quickly test different approaches to explaining complex policies and see which ones produce the most clear and helpful responses for new employees.

# %%
async def get_completion(prompt, system_prompt="You are a helpful assistant.", temperature=0.7):
    """Get a completion from the OpenAI API"""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",  # You can change this to your preferred model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content

# %%
response = await get_completion(input())
display.Markdown(response)

# %% [markdown]
# ## 4. Prompt Scenarios: HR Policy Explanation
#
# Now let's explore a prompt scenario relevant to our HR onboarding use case. We'll create prompts for an AI assistant that explains company policies to new employees.
#
# We'll first try a basic prompt without much engineering, then create an improved version using the principles we've discussed. This will demonstrate how proper prompt engineering can significantly improve the quality of policy explanations for new employees.

# %%
# Let's use our HR policy explanation example instead of the research paper
hr_policy = """
Company Work-From-Home Policy

Effective Date: January 1, 2023

1. Policy Purpose
This policy establishes guidelines for employees to work from home while maintaining productivity and effective operations.

2. Eligibility
- Full-time employees who have completed their three-month probationary period
- Employees with satisfactory performance reviews
- Roles suitable for remote work as determined by department managers

3. Work-From-Home Arrangements
- Employees may work from home up to two (2) days per week
- Requests must be submitted through the HR portal at least 24 hours in advance
- Approval is at the discretion of the employee's direct manager
- Regular work hours (9am-5pm) must be maintained during remote work days

4. Employee Responsibilities
- Maintain availability during business hours via approved communication channels
- Attend all scheduled meetings virtually
- Ensure a secure internet connection and appropriate workspace
- Follow all data security and confidentiality requirements
- Report any work-related accidents that occur during remote work hours

5. Equipment and Support
- Company will provide laptop and necessary software
- IT support available during normal business hours
- Employees are responsible for maintaining reliable internet connection

6. Termination of Arrangement
The company reserves the right to modify or terminate work-from-home arrangements if:
- Business needs change
- Performance or productivity concerns arise
- Policy violations occur
"""

# Basic prompt without engineering techniques
basic_prompt = f"Explain this work-from-home policy: {hr_policy}"

# Engineered prompt using best practices for HR onboarding
engineered_prompt = f"""
You are an HR specialist helping new employees understand company policies during their onboarding process.

Please explain the following work-from-home policy in a clear, friendly manner that new employees will find helpful and easy to understand. 

Format your response with these components:
1. A brief title and overview
2. Clear sections with headings covering eligibility, how to request WFH days, and employee responsibilities
3. A Q&A section addressing 3 common questions new employees might have
4. Contact information for additional help

Use bullet points and numbered lists where appropriate to make the information easily scannable. 
Maintain a friendly, supportive tone appropriate for new employees who are still learning company procedures.

POLICY TO EXPLAIN:
{hr_policy}
"""

# %%
basic_response = await get_completion(basic_prompt)

print("==== BASIC PROMPT RESPONSE ====\n")
display.Markdown(basic_response)

# %%
engineered_response = await get_completion(engineered_prompt)

print("==== ENGINEERED PROMPT RESPONSE ====\n")
display.Markdown(engineered_response)

# %% [markdown]
# ### Comparing the Responses
#
# Notice the significant differences between the two responses:
#
# **Basic Prompt Response:**
# - Presents information as a dense paragraph
# - Lacks clear structure and organization
# - Uses more formal, policy-like language
# - Misses some details from the original policy
# - Difficult for new employees to quickly scan and understand
#
# **Engineered Prompt Response:**
# - Uses a clear, hierarchical structure with headings
# - Incorporates bullet points and numbering for easier scanning
# - Adds a helpful Q&A section addressing common concerns
# - Uses a friendly, supportive tone appropriate for onboarding
# - Includes contact information for further questions
# - Much more useful for new employees learning about company policies
#
# This example demonstrates how proper prompt engineering can significantly improve policy explanations for employee onboarding, making complex information more accessible and user-friendly.

# %% [markdown]
# ## 5. Few-Shot Learning Example: HR Case Handling
#
# Next, let's create a prompt for an HR assistant that handles employee questions about policies using few-shot examples. Few-shot learning is particularly valuable in HR contexts because it helps ensure consistent handling of policy questions across the organization.

# %%
# Here we'll set up an HR policy question handler using few-shot examples
hr_questions = """
1. What types of personal leave are available?
2. How do I submit a request for professional development funding?
3. What is the process for reporting workplace harassment?
"""


# %%
# Few-shot HR assistant prompt
def create_hr_assistant_prompt(hr_questions):
    hr_assistant_prompt = f"""
You are an experienced HR specialist who provides clear, helpful responses to employee questions about company policies.
Your goal is to give accurate information in an approachable, easy-to-understand format.

Here are examples of the kind of responses you should provide:

QUESTION 1:
How does the company's health insurance work?

RESPONSE 1:
Our company offers two health insurance plans:

• Standard Plan: $500 deductible, 80% coverage after deductible, $25 copay for office visits
• Premium Plan: $250 deductible, 90% coverage after deductible, $15 copay for office visits

Enrollment periods:
1. When you're first hired (within 30 days)
2. During open enrollment (November 1-15 each year)
3. After qualifying life events (marriage, birth of child, etc.)

Coverage begins on the 1st day of the month following enrollment.

For detailed plan documents or questions about specific coverage, contact benefits@company.com.

QUESTION 2:
What is the company's policy on performance reviews?

RESPONSE 2:
Performance reviews are conducted twice yearly:

• Mid-year review (June): Informal check-in on goals, development, and progress
• Annual review (December): Formal evaluation tied to compensation decisions

The process works like this:
1. You'll complete a self-assessment in the HR system
2. Your manager will complete their assessment of your performance
3. You'll have a one-on-one meeting to discuss feedback and set new goals
4. Both you and your manager will sign off on the final review

New employees hired less than 3 months before a review period may have a modified process.

For specific questions about your upcoming review, please speak with your direct manager.

Now, please respond to the following employee questions in the same clear, helpful format:

{hr_questions}
"""
    return hr_assistant_prompt

hr_assistant_prompt = create_hr_assistant_prompt(hr_questions)

hr_responses = await get_completion(hr_assistant_prompt)
display.Markdown(hr_responses)


# %% [markdown]
# ### The Power of Few-Shot Learning for HR Policy Explanation
#
# The example above demonstrates how few-shot learning can dramatically improve the quality of HR policy explanations. By providing examples of well-structured, clear responses to policy questions, we:
#
# 1. **Ensure consistency** in how policies are explained to employees
# 2. **Establish a standardized format** that makes information easy to scan and understand
# 3. **Model the appropriate tone** that balances professionalism with accessibility
# 4. **Include practical details** that employees need, like contact information and next steps
#
# This approach is particularly valuable during employee onboarding, where new team members are encountering many company policies for the first time and need clear, actionable information.

# %% [markdown]
# ## 6. Chain-of-Thought Prompting: Complex Policy Explanation
#
# Now let's explore chain-of-thought prompting for explaining a more complex HR policy. This technique is especially useful for policies with multiple conditions, exceptions, or procedural steps that need to be clearly explained to new employees.

# %%
# Complex travel and expense policy
complex_policy = """
COMPANY TRAVEL AND EXPENSE REIMBURSEMENT POLICY

1. GENERAL GUIDELINES
1.1 This policy applies to all employees traveling for business purposes and submitting expenses for reimbursement.
1.2 All business travel requires manager approval prior to booking.
1.3 Employees must book travel through the designated corporate travel portal.
1.4 All expenses must be reasonable, necessary, and for legitimate business purposes.

2. TRANSPORTATION
2.1 Air Travel
   2.1.1 Economy class is required for flights under 8 hours duration.
   2.1.2 Business class is permitted for flights over 8 hours duration.
   2.1.3 Tickets should be purchased at least 14 days in advance when possible.

2.2 Ground Transportation
   2.2.1 Personal vehicle use is reimbursed at $0.55 per mile.
   2.2.2 Rental cars must be economy or compact class unless 3+ employees are traveling together.
   2.2.3 Standard ride services are permitted; premium options are not reimbursable.

3. LODGING
3.1 Employees should stay at preferred partner hotels when available.
3.2 Maximum nightly rates vary by location and are specified in the travel portal.
3.3 Standard single rooms are authorized; upgrades are at employee's expense.

4. MEALS AND ENTERTAINMENT
4.1 Meals are reimbursed at actual cost with appropriate receipts.
4.2 Daily meal limits are: $60 domestic, $100 international.
4.3 Itemized receipts are required for all purchases over $25.
4.4 Client entertainment requires pre-approval and business justification.

5. EXPENSE REPORTING
5.1 All expense reports must be submitted within 15 days of trip completion.
5.2 Reports must be submitted through the expense management system.
5.3 Original or digital copies of receipts must be attached for all expenses over $25.
5.4 Expenses are typically reimbursed within 7-10 business days after approval.

6. NON-REIMBURSABLE ITEMS
6.1 Personal entertainment (movies, gym fees, etc.)
6.2 Expenses for spouse or family members
6.3 Flight upgrades outside of policy
6.4 Alcohol (except for client entertainment with prior approval)

7. EXCEPTIONS
7.1 Exceptions to this policy require CFO approval for expenses exceeding limits by more than 10%.
7.2 Exception requests must be submitted with written justification.
"""

# Standard prompt
standard_explanation_prompt = f"""
Explain this travel and expense reimbursement policy to new employees:

{complex_policy}
"""

# Chain-of-thought prompt
cot_explanation_prompt = f"""
You are an HR specialist helping new employees understand company policies during onboarding.

Please explain the following travel and expense policy in a way that's easy for new employees to understand and follow. 

To provide a comprehensive explanation, please:

1. First, explain the basic purpose of the policy and who it applies to.
2. Then, break down the process into logical steps (before travel, during travel, after travel).
3. For each step, explain what the employee needs to do, what limits apply, and any exceptions.
4. Highlight common mistakes new employees make with expense reporting.
5. Finally, provide information on who to contact with questions.

Use a friendly, helpful tone and format your response with clear headings, bullet points, and numbered lists where appropriate.
Title your response "Travel & Expense Policy Explanation for New Employees"

POLICY TO EXPLAIN:
{complex_policy}
"""

standard_explanation = await get_completion("How much will I be reimbursed for international travel?", standard_explanation_prompt)
cot_explanation = await get_completion("How much will I be reimbursed for international travel?", cot_explanation_prompt)

print("==== STANDARD EXPLANATION ====\n")
print(standard_explanation)
print("\n\n==== CHAIN-OF-THOUGHT EXPLANATION ====\n")
print(cot_explanation)

# %% [markdown]
# ### Benefits of Chain-of-Thought Prompting for Complex Policies
#
# Notice how the chain-of-thought approach transforms a complex policy into a step-by-step guide that's much easier for new employees to understand and follow. Key advantages include:
#
# 1. **Logical Flow**: The policy is explained as a process with clear steps, making it easier to understand the sequence of actions required.
#
# 2. **Contextual Understanding**: Each section explains not just what to do, but why it matters and how it fits into the bigger picture.
#
# 3. **Practical Focus**: The explanation emphasizes what employees need to know to comply with the policy correctly.
#
# 4. **Anticipating Questions**: By addressing common mistakes and special circumstances, the explanation proactively answers questions new employees are likely to have.
#
# This approach is particularly valuable for complex HR policies that might otherwise be overwhelming during the onboarding process.

# %% [markdown]
# ## 7. Evaluation with Azure AI Evaluation SDK
#
# Now let's explore Microsoft's Azure AI Evaluation SDK to quantitatively evaluate our prompts. For HR policy explanations during onboarding, it's important to measure how clear, helpful, and accurate the explanations are - evaluation helps ensure our prompts are genuinely helping new employees understand company policies.

# %%
# Import the Azure AI Evaluation SDK
from azure.ai.evaluation import RelevanceEvaluator, FluencyEvaluator, CoherenceEvaluator

# Configure the model for evaluation
model_config = {
    "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "api_key": os.getenv("AZURE_OPENAI_KEY"),
    "azure_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    "api_version": "2024-12-01-preview"
}

# %% [markdown]
# ### Evaluating HR Policy Explanations

# %%
# Initialize evaluators
relevance_evaluator = RelevanceEvaluator(model_config)
fluency_evaluator = FluencyEvaluator(model_config)
coherence_evaluator = CoherenceEvaluator(model_config)

# Evaluate basic summary
basic_relevance = relevance_evaluator(
    query="Summarize this research paper on deep reinforcement learning for autonomous driving",
    ground_truth=research_paper,
    response=basic_summary
)

basic_fluency = fluency_evaluator(
    response=basic_summary
)

basic_coherence = coherence_evaluator(
    conversation={
        "messages": [
            {"role": "user", "content": "Summarize this research paper on deep reinforcement learning for autonomous driving"},
            {"role": "assistant", "content": basic_summary}
        ]
    }
)

# Evaluate engineered summary
engineered_relevance = relevance_evaluator(
    query="Summarize this research paper on deep reinforcement learning for autonomous driving with a structured format",
    ground_truth=research_paper,
    response=engineered_summary
)

engineered_fluency = fluency_evaluator(
    response=engineered_summary
)

engineered_coherence = coherence_evaluator(
    conversation={
        "messages": [
            {"role": "user", "content": "Summarize this research paper on deep reinforcement learning for autonomous driving with a structured format"},
            {"role": "assistant", "content": engineered_summary}
        ]
    }
)

# Print evaluation results
print("Basic Prompt Evaluation:")
print(f"- Relevance: {basic_relevance}")
print(f"- Fluency: {basic_fluency}")
print(f"- Coherence: {basic_coherence}")
print("\nEngineered Summary Evaluation:")
print(f"- Relevance: {engineered_relevance}")
print(f"- Fluency: {engineered_fluency}")
print(f"- Coherence: {engineered_coherence}")


# %% [markdown]
# ### Visualizing Evaluation Results
#
# Visualizing evaluation results helps us understand the strengths and weaknesses of different prompting approaches. For HR policy explanations, this data can help determine which approaches are most effective for helping new employees understand complex policies.

# %%
def visualize_evaluation(basic_scores, engineered_scores, metrics):
    """Create a bar chart comparing evaluation scores"""
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, basic_scores, width, label='Basic Prompt')
    ax.bar(x + width/2, engineered_scores, width, label='Engineered Prompt')
    
    ax.set_ylim(0, 5)
    ax.set_ylabel('Score')
    ax.set_title('Prompt Evaluation Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

# Extract scores
basic_scores = [basic_relevance.get('score', 0), basic_fluency.get('score', 0), basic_coherence.get('score', 0)]
engineered_scores = [engineered_relevance.get('score', 0), engineered_fluency.get('score', 0), engineered_coherence.get('score', 0)]
metrics = ['Relevance', 'Fluency', 'Coherence']

visualize_evaluation(basic_scores, engineered_scores, metrics)

# %% [markdown]
# ## 8. Batch Evaluation with Multiple Prompts
#
# Now let's evaluate multiple prompts using the Azure AI Evaluation SDK's batch evaluation capabilities:

# %%
from azure.ai.evaluation import evaluate
import json

# Create a sample dataset for evaluation
evaluation_data = [
    {
        "query": "Summarize this research paper",
        "ground_truth": research_paper,
        "response_basic": basic_summary,
        "response_engineered": engineered_summary
    },
    {
        "query": "Review this Python code",
        "ground_truth": code_to_review,
        "response_basic": await get_completion(f"Review this Python code: {code_to_review}"),
        "response_engineered": code_review
    },
    {
        "query": "Analyze this sales data",
        "ground_truth": data_table,
        "response_basic": standard_analysis,
        "response_engineered": cot_analysis
    }
]

# Write data to a JSONL file
with open('evaluation_data.jsonl', 'w') as f:
    for item in evaluation_data:
        f.write(json.dumps(item) + '\n')

# Set up evaluators
evaluators = {
    "relevance": relevance_evaluator,
    "coherence": coherence_evaluator,
    "fluency": fluency_evaluator
}

# Run evaluation for basic responses
basic_results = evaluate(
    data="evaluation_data.jsonl",
    evaluators=evaluators,
    evaluator_config={
        "default": {
            "column_mapping": {
                "query": "${data.query}",
                "ground_truth": "${data.ground_truth}",
                "response": "${data.response_basic}"
            }
        }
    },
    output_path="basic_evaluation_results.json"
)

# Run evaluation for engineered responses
engineered_results = evaluate(
    data="evaluation_data.jsonl",
    evaluators=evaluators,
    evaluator_config={
        "default": {
            "column_mapping": {
                "query": "${data.query}",
                "ground_truth": "${data.ground_truth}",
                "response": "${data.response_engineered}"
            }
        }
    },
    output_path="engineered_evaluation_results.json"
)

# Display results
print("Basic Prompt Evaluation Summary:")
for metric, value in basic_results.get('summary').items():
    print(f"{metric}: {value}")

print("\nEngineered Prompt Evaluation Summary:")
for metric, value in engineered_results.get('summary').items():
    print(f"{metric}: {value}")

# %% [markdown]
# ## 9. Key Takeaways and Best Practices
#
# Based on our experiments with prompt engineering and evaluation, here are the key takeaways:
#
# 1. **Clear Role Definition**: Defining the AI's role significantly improves response quality
# 2. **Structured Formatting**: Requesting specific output formats produces more organized, usable outputs
# 3. **Few-Shot Examples**: Providing examples dramatically improves the model's ability to follow patterns
# 4. **Chain-of-Thought**: Guiding the model through a reasoning process improves analytical outputs
# 5. **Quantitative Evaluation**: Using consistent metrics helps identify areas for improvement
# 6. **Multiple Metrics**: Evaluating along different dimensions provides a more complete picture of prompt quality
#
# These principles can be applied to any prompt engineering task and help create more effective AI interactions.

# %% [markdown]
# ## 10. Conclusion
#
# In this challenge, we've explored fundamental prompt engineering techniques and evaluation methods using Microsoft's Azure AI Evaluation SDK. We've seen how different prompting approaches can dramatically improve the quality, relevance, and usefulness of AI-generated content.
#
# By applying structured evaluation metrics, we were able to quantitatively measure improvements and guide our prompt refinement process. This systematic approach to prompt engineering can be applied to many different business scenarios, helping to create more effective AI interactions.
