from langchain_google import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, max_completion_tokens=100)
result = model.invoke("Write a poem about Nepal.")
print(result.content)
