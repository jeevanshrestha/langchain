from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()
import os
# This will automatically use the HUGGINGFACEHUB_API_TOKEN env var
chat_model = ChatHuggingFace.from_huggingface_api(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # or TinyLlama/TinyLlama-1.1B-Chat-v1.0
    model_kwargs={"temperature": 0.5, "max_new_tokens": 100},
)

response = chat_model.invoke("What is the capital of India?")
print(response.content)
