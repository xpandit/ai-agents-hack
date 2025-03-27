# Challenge 3: Fundamentals of Prompt Engineering & Evaluation

In this challenge, you'll master the art of writing effective prompts, explore interactive development techniques, and learn how to evaluate prompt quality using Microsoft's tools.

## Objectives

- Understand the principles of effective prompt engineering
- Learn how prompt development and evaluation works in practice
- Explore Microsoft's prompt evaluation frameworks
- See examples of prompts for specific agent tasks
- Understand how prompt performance can be improved through systematic evaluation

## What You'll Learn

### 1. Prompt Engineering Fundamentals
The notebook explains key principles of effective prompt engineering:
- Role & context setting
- Clear instructions
- Input/output format specification
- Few-shot examples
- Chain-of-thought prompting
- System vs User prompts

### 2. Interactive Prompt Development
The notebook demonstrates:
- Setting up a prompt testing environment
- Iterative prompt refinement techniques
- Visualization of prompt improvements

### 3. Prompt Scenarios & Examples
You'll explore real-world examples of prompts for:
- A research assistant that summarizes academic papers
- A code reviewer that provides feedback on code quality
- A data analyst that extracts insights from tables

### 4. Evaluation Methods
Learn about different approaches to evaluate your prompts:
- Microsoft's Prompt Flow framework
- Custom evaluation metrics
- A/B testing different prompt versions

### 5. Optimization Techniques
See practical examples of prompt optimization:
- Before/after comparisons
- Quantified improvements
- Best practices for systematic improvement

## Getting Started

This challenge comes with a pre-prepared Jupyter notebook (`3-prompt-fundamentals.ipynb`) that walks you through various aspects of prompt engineering and evaluation. **No coding required** - the notebook is ready to run, with clear explanations and examples at each step.

### Verify your resources' creation

Go back to your `Azure Portal` and find your `Resource Group` that should by now contain 2 resources and look like this:

![image](https://github.com/user-attachments/assets/e04298dd-a601-47a2-8fda-bd0cac19f313)

After checking the deployed resources, you will need to configure the environment variables in the `.env` file. The `.env` file is a configuration file that contains the environment variables for the application. The `.env` file is automatically created running the following command within the terminal in your Codespace:

```bash
cd challenge-3
./import.sh --resource-group <resource-group-name>
```

This script will connect to Azure and fetch the necessary keys and populate the `.env` file with the required values in the root directory of the repository. If needed, the script will prompt you to sign in to your Azure account.

When connecting to Azure, you will be asked to choose which subscription to use.

### Verify `.env` setup

When the script is finished, review the `.env` file to ensure that all the values are correct. If you need to make any changes, you can do so manually.

The default sample has an `.env.sample` file that shows the relevant environment variables that need to be configured in this project. The script should create a `.env` file that has these same variables _but populated with the right values_ for your Azure resources.

If the file is not created, simply copy over `.env.sample` to `.env` - then populate those values manually from the respective Azure resource pages using the Azure Portal.

## How to Complete This Challenge

1. Open the [3-prompt-fundamentals.ipynb](3-prompt-fundamentals.ipynb) notebook
2. Run through each cell in sequence
3. Read the explanations and observe the examples
4. Try modifying some of the example prompts to see how changes affect outputs
5. Complete any exercises marked as "Your Turn" in the notebook

## Resources

- [Microsoft's Prompt Engineering Guide](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering)
- [Prompt Flow Documentation](https://microsoft.github.io/promptflow/)
- [LangChain Evaluation Documentation](https://python.langchain.com/docs/guides/evaluation/)

## Next Steps

After completing this challenge, you'll have a solid understanding of prompt engineering principles and evaluation techniques that will be essential for the upcoming challenges focused on building AI agents with tools and multi-agent systems.

Good luck!