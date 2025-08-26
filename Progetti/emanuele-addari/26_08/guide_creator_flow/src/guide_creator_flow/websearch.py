import os                                                                                                                                   
from crewai import Agent, Task, Crew
# Importing crewAI tools
from crewai_tools import (
    WebsiteSearchTool
)
from dotenv import load_dotenv
load_dotenv()
# Set up API keys

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Instantiate tools
web_rag_tool = WebsiteSearchTool()

# Create agents
researcher = Agent(
    role='Website Research Analyst',
    goal='Provide up-to-date web analysis',
    backstory='An expert analyst',
    tools=[web_rag_tool],
    verbose=True
)

writer = Agent(
    role='Content Writer',
    goal='Craft engaging summaries about the web results',
    backstory='A skilled writer.',
    verbose=True
)

research = Task(
    description='Look up the query on the web and retrieve the first 5 results.',
    expected_output='The top 5 results',
    agent=researcher
)

write = Task(
    description='Write a detailed summary about the web search results.',
    expected_output='A 4-paragraph blog post formatted in markdown with engaging, informative, and accessible content, avoiding complex jargon.',
    agent=writer,
    output_file='blog-posts/new_post.md'  # The final blog post will be saved here
)

# Assemble a crew with planning enabled
crew = Crew(
    agents=[researcher, writer],
    tasks=[research, write],
    verbose=True,
    planning=True
)

# Execute tasks
crew.kickoff()