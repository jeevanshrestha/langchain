from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.0)

result = model.invoke("Write a poem about Nepal.")
print(result.content)