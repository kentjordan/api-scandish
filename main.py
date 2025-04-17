from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_together import ChatTogether
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage
from fastapi.responses import JSONResponse

load_dotenv()
app = FastAPI()

model = ChatTogether(model="meta-llama/Llama-4-Scout-17B-16E-Instruct")


class Recipe(BaseModel):
    title: str
    content: str


parser = PydanticOutputParser(pydantic_object=Recipe)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            "You're expert and good in making Filipino food recipes and you will provide recipe based on the user given input."
        ),
        SystemMessage(
            "You're about to response in JSON format, the instruction format is given below."
        ),
        SystemMessage(parser.get_format_instructions()),
        ("human", "{food}"),
    ]
)

chain = prompt | model | parser


@app.post("/generate")
async def generate():
    recipe = chain.invoke({"food": "mang inasal"}).model_dump_json()
    return JSONResponse(recipe)


@app.get("/")
def root():
    return "hey!"
