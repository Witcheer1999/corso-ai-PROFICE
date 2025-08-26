import random
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel
from crewai import LLM

class ExampleState(BaseModel):
    choice: str = ""
    response: str = ""

class RouterFlow(Flow[ExampleState]):

    @start()
    def start_method(self):
        print("Starting the flow")
        llm = LLM(model="azure/gpt-4o")
        
        # Randomly decide to generate either a city or a country
        self.state.choice = random.choice(["city", "country"])
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output only the name of a city or country, nothing else."},
            {"role": "user", "content": f"Generate the name of a random {self.state.choice}. Only output the name."}
        ]

        # Make the LLM call with JSON response format
        self.state.response = llm.call(messages=messages)
        
    @router(start_method)
    def select_method(self):
        if self.state.choice == "city":
            return "city"
        else:
            return "country"

    @listen("city")
    def generate_fact(self):
        llm = LLM(model="azure/gpt-4o")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output a fun fact about a city."},
            {"role": "user", "content": f"Generate a fun fact about {self.state.response}."}
        ]

        # Make the LLM call with JSON response format
        print(llm.call(messages=messages))

    @listen("country")
    def generate_neighbors(self):
        llm = LLM(model="azure/gpt-4o")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output the neighboring countries of a given country."},
            {"role": "user", "content": f"Generate the list of neighbours for the country {self.state.response}."}
        ]
        
        print(llm.call(messages=messages))


def kickoff():
    flow = RouterFlow()
    flow.plot("exercise_flow_plot")
    flow.kickoff()
    
if __name__ == "__main__":
    kickoff()