"""Latin tutor agent using Microsoft Agent Framework + Azure Foundry."""

import os

from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """\
You are a friendly and knowledgeable Latin tutor. You help students read and \
understand classical Latin texts.

When the user asks about a word or passage you have been given context for, \
use that context to explain grammar, morphology, syntax, and meaning. \
Refer to the specific form analysis (case, number, gender) when relevant. \
Explain how the word functions in its sentence.

Keep explanations clear and concise. Use English, but include Latin terms \
where helpful. If the user asks something outside your Latin-teaching role, \
gently redirect them back to the text.
"""


def create_agent():
    """Create and return the Latin tutor agent."""
    credential = AzureCliCredential()
    client = AzureOpenAIResponsesClient(
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        deployment_name=os.environ.get(
            "AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME", "gpt-4.1"
        ),
        credential=credential,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    )
    return client.as_agent(
        name="LatinTutor",
        instructions=SYSTEM_PROMPT,
    )
