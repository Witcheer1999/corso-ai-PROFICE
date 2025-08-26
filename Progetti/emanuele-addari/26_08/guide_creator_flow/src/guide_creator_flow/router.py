import random
import os
from crewai.flow.flow import Flow, listen, router, start
from crewai import LLM
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

random_choice = "Randomly choose city or country. Absolutely only say city or country"
choice_prompt = "If you get city, choose a random one and say a fun fact about it. If you get country, choose a random country and say the neighboring countries."

class SimpleLLM:
    def __init__(self):
        model = os.getenv("MODEL", "azure/gpt-4.1")
        self.llm = LLM(model=model)

    def get_response(self, input):
        messages = [
            {"role": "user", "content": f"{input}"}
        ]
        response = self.llm.call(messages=messages)
        return response

class ExampleState(BaseModel):
    flag: bool = False
    choice: str = ""

class RouterFlow(Flow[ExampleState]):

    @start()
    def RandomChoice(self):
        print("Starting the structured flow")
        llm_choice = SimpleLLM().get_response(random_choice)
        self.state.choice = llm_choice.strip().lower()
        self.state.flag = "city" in self.state.choice
        print(f"LLM chose: {self.state.choice}")
        print(f"Flag set to: {'city' if self.state.flag else 'country'}")
        return self.state

    @router(RandomChoice)
    def Splitter(self):
        if self.state.flag:
            return "city"
        else:
            return "country"

    @listen("city")
    def CityFunFact(self):
        result = SimpleLLM().get_response("Choose a random city and say a fun fact about it")
        print(f"City result: {result}")
        return result

    @listen("country")
    def NeighboringCountries(self):
        result = SimpleLLM().get_response("Choose a random country and say the neighboring countries")
        print(f"Country result: {result}")
        return result

def run_test():
    flow = RouterFlow()
    flow.plot("my_flow_plot")
    result = flow.kickoff()
    return result

if __name__ == "__main__":
    run_test()