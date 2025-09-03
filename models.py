from langchain.chat_models import init_chat_model
from langchain_together import ChatTogether
from langchain_ollama import ChatOllama


# Initialize model
def init_model():
    # init_chat_model(model="openai:gpt-4.1", temperature=0.0)

    # choose from our 50+ models here: https://docs.together.ai/docs/inference-models
    # ChatTogether(
    #     # model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    #     model="lgai/exaone-3-5-32b-instruct",
    # )

    return ChatOllama(
        model="llama3.1:8b",
        # model="gemma3:1b",
        # temperature=0.2,
    )