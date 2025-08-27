from crewai.flow.flow import Flow, listen, start, router
from crewai import LLM
import os
from web_search_crew.crews.web_search_crew.web_search_crew import WebSearchCrew
from pydantic import BaseModel



class GuideCreatorState(BaseModel):
    topic: str = ""
    ethics_analysis: str = ""

class GuideCreatorFlow(Flow[GuideCreatorState]):
    """Flow for creating a comprehensive guide on any topic"""

    @start() # primo step utile per task non lineari
    def get_user_input(self):
        """Get input from the user about the guide topic and audience"""
        print("\n=== Create Your Comprehensive Guide ===\n")

        # Get user input
        self.state.topic = input("What topic would you like to search? ")

        print(f"\nCreating a report on {self.state.topic}\n")
        return self.state

    @listen(get_user_input)
    def ethics_checker(self, state):
        """Check the ethical considerations for the guide topic"""
        print("Checking ethical considerations...")

        # Initialize the LLM
        llm = LLM(model="azure/gpt-4.1") 

        # Create the messages for the outline
        messages = [
            {"role": "system", "content": "You are an assistant specialized in evaluating the ethical implications of topics. Analyze the given topic for potential ethical issues, risks, or sensitivities. Output your analysis in JSON format, clearly indicating any ethical concerns and suggestions for mitigation."},
            {"role": "user", "content": f"""
            Analyze the topic \"{state.topic}\" for ethical implications, risks, and sensitivities. Respond only with a string: 'positive' if the topic is ethically acceptable. If there are significant ethical concerns, respond with 'negative' followed by a brief explanation of the reason.
            """},
        ]
        response = llm.call(messages=messages)
        # print(f"LLM response: {response}")
        self.state.ethics_analysis = str(response)
        print(f"Ethics analysis result: {self.state.ethics_analysis}")
        return self.state


    @router(ethics_checker)
    def ethics_result(self):
        if self.state.ethics_analysis == "positive":
            print("Ethics analysis passed.")
            # Ensure output directory exists before saving
            os.makedirs("output", exist_ok=True)
            return "passed"
        else:
            print("Ethics analysis failed.")
            print(f"Ethics analysis details: {self.state.ethics_analysis}")
            return "failed"

    @listen("passed")
    def run_crew(self):
        """Run the web search crew to gather information and create the guide"""
        print("Running the web search crew...")

        # Initialize and run the crew
        crew = WebSearchCrew()
        crew_output = crew.crew().kickoff(
            inputs={
                "topic": self.state.topic,
                "web_search_task": {
                    "topic": self.state.topic
                },
                "summarization_task": {
                    "topic": self.state.topic
                }
            }
        )

        print("Web search crew completed.")
        # print(crew_output)
       

    @listen("failed")
    def handle_failed_ethics(self):
        """Handle failed ethics check"""
        print("Ethics check failed. Please choose a different topic.")
        return "failed"

def kickoff():
    """Start the guide creator flow"""
    GuideCreatorFlow().kickoff()
    print("\n=== Flow Complete ===")
    print("Your comprehensive guide is ready in the output directory.")
    print("Open output/report.md to view it.")

def plot():
    """Generate a visualization of the flow"""
    flow = GuideCreatorFlow()
    flow.plot("web_search_crew_flow")
    print("Flow visualization saved to web_search_crew_flow.html")

if __name__ == "__main__":
    kickoff()