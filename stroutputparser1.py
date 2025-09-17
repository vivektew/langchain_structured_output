from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

# 1st Prompt detailed Report
template1 = PromptTemplate(
    template="Write a detailed report on topic: {topic}",
    input_variables=["topic"]
)

# 2nd Prompt to give summary
template2 = PromptTemplate(
    template="Write a 5 lines summary on text: /n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'black holes'})

print(result)