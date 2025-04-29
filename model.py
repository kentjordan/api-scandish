from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_together import ChatTogether
from pydantic import BaseModel
from torch import nn
from torchvision.models import densenet201
import torch

load_dotenv()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

vision_model = densenet201()
vision_model.classifier = nn.Linear(vision_model.classifier.in_features, 5)

vision_model.load_state_dict(torch.load("./model.pt"))
vision_model = vision_model.to(device)

# ----------------- LLM ---------------------- #
llm_model = ChatTogether(model="meta-llama/Llama-4-Scout-17B-16E-Instruct")

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

llm = prompt | llm_model | parser