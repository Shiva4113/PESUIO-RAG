from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

search_tool = SerperDevTool()

@CrewBase
class AnimeSearchCrew():
    """Crew for searching anime merchandise"""

    @agent
    def anime_researcher(self) -> Agent:
        """Creates the anime merchandise researcher agent"""
        return Agent(
            config=self.agents_config['anime_researcher'],
            tools=[search_tool],
            verbose=True
        )

    @agent
    def price_analyzer(self) -> Agent:
        """Creates the price analyzer agent"""
        return Agent(
            config=self.agents_config['price_analyzer'],
            tools=[search_tool],
            verbose=True
        )

    @task
    def research_merchandise(self) -> Task:
        """Task for researching merchandise"""
  
        return Task(
            config=self.tasks_config['research_merchandise']
        )

    @task
    def analyze_prices(self) -> Task:
        """Task for analyzing prices"""
        return Task(
            config=self.tasks_config['analyze_prices'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the anime merchandise search crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )