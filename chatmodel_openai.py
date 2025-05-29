from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4" , temperature=0.0, max_completion_tokens=100)

result = model.invoke("Write a poem about Nepal.")

print(result.content)

