from typing import Dict, List, Optional, Tuple
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from social_world_model.agents import LLMAgent
from social_world_model.database import Observation
from sotopia.generation_utils import agenerate
from typing import Any

class ObsDistribution(BaseModel):
    agents_per_observation: List[Tuple[str, List[str]]] = Field(description="The list of observations with a list of agents that perceive the observation")

class FormattedQuestion(BaseModel):
    agent_name: str = Field(description="The name of the agent that the question should be directed to")
    question_observation: Observation = Field(description="The observation containing the question to be asked to the agent")

class Simulation(BaseModel):
    agents: List[str] = Field(description="The list of agents in the simulation")
    agent_memories: Dict[str, List[str]] = Field(description="The list of memories for each agent")
    question: str = Field(description="The question that was asked to the agent")
    reasoning: str = Field(description="The reasoning that the agent did to answer the question")
    answer: str = Field(description="The answer that the agent gave to the question")

class ToMEngine:
    def __init__(self, agent_prompt: str = "", model_name: str = "gpt-4o-mini") -> None:
        self.agents: Dict[str, LLMAgent] = {}
        self.agent_prompt: str = agent_prompt # The overall prompt for the agents to follow
        self.current_time = 0
        self.model_name = model_name
        self.simulation: Simulation = Simulation(
            agents=[],
            agent_memories={},
            question="",
            reasoning="",
            answer=""
        )

    def add_agent(self, name: str) -> None:
        self.agents[name] = LLMAgent(
            name, 
            agent_prompt=self.agent_prompt,
            model_name=self.model_name
        )

    def reset_agents(self) -> None:
        self.agents = {}
        self.current_time = 0

    def get_simulation(self) -> Simulation:
        self.simulation.agents = list(self.agents.keys())
        self.simulation.agent_memories = {agent: 
        [message.last_turn for message in self.agents[agent].message_history] for agent in self.agents}
        return self.simulation
        
    async def distribute_observations(self, event: str) -> List[Tuple[str, List[str]]]:
        agent_memory = await agenerate(
            model_name=self.model_name,
            template=(
                "Process the following event and generate a list of agent names and their observations.\n"
                "Event: {event}\n\n"
                "Simply extract who did what"
            ),
            input_values={
                "event": event,
            },
            temperature=0.3,
            output_parser=PydanticOutputParser(pydantic_object=ObsDistribution),
            )
        print(agent_memory)
        assert isinstance(agent_memory, ObsDistribution), "Output parser did not return an ObsDistribution"
        return agent_memory.agents_per_observation
    
    async def initialize_simulation(self, agent_names: list[str], observations_with_perceivers: List[Tuple[str, List[str]]]) -> None:
        for agent_name in agent_names:
            self.add_agent(agent_name)
        for obs, perceivers in observations_with_perceivers:
            for agent in perceivers:
                self.agents[agent].message_history.append(
                    Observation(
                        agent_name=agent,
                        last_turn=obs,
                        turn_number=self.current_time,
                        available_actions=["none", "speak", "non-verbal communication", "action", "leave"]
                    )
                )

    async def reason_about_belief(self, question: str, agents: list[str], target_agent: str|None=None, answer_candidates: list[str]|None=None) -> Tuple[str, str]:
        if not target_agent:
            formatted_question = await agenerate(
                model_name=self.model_name,
                template="Please reformat the following question: {question}. Change the question from third person perspective to a second person perspective as if an interviewer is asking the question. The question should be directed to one of the following agents: {agents}",
                input_values={
                    "question": question,
                    "agents": agents,
                },
                temperature=0.3,
                output_parser=PydanticOutputParser(pydantic_object=FormattedQuestion),
            )
            print(formatted_question)
            target_agent = formatted_question.agent_name
            question_observation = formatted_question.question_observation
            if answer_candidates:
                question_observation.last_turn += f"The question should be answered by one of the following candidates: {answer_candidates}"
        else:
            question_observation = Observation(
                agent_name=target_agent,
                last_turn=question,
                turn_number=self.current_time,
                available_actions=["none", "speak", "non-verbal communication", "action", "leave"]
            )
            if answer_candidates:
                question_observation.last_turn += f"The question should be answered by one of the following candidates: {answer_candidates}"
        assert target_agent in self.agents, f"Agent {target_agent} not found in agents"
        action = await self.agents[target_agent].aact(question_observation)
        assert action is not None, f"Action is None for {question_observation.last_turn}"
        reasoning_and_answer = action.argument
        try:
            reasoning = reasoning_and_answer.split("<reasoning>")[1].split("</reasoning>")[0]
            answer = reasoning_and_answer.split("<answer>")[1].split("</answer>")[0]
        except Exception as e:
            reasoning = ""
            answer = reasoning_and_answer
        self.simulation.reasoning = reasoning
        self.simulation.answer = answer
        self.simulation.question = question
        return reasoning, answer
