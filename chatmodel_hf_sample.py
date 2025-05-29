from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.schema import HumanMessage

# Load model and tokenizer locally
model_name = "tiiuae/falcon-7b-instruct"  # or any other compatible causal model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Build transformers pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
)

# Wrap in LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Use it with LangChain schema
response = llm.invoke("Explain what LangChain is.")
print(response)
