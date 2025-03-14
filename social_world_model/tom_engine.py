from typing import Dict, List, Tuple
from sotopia.generation_utils import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from social_world_model.agents import LLMAgent
from social_world_model.database import Observation, SocializedContext
from sotopia.generation_utils import agenerate
from typing import Any
import json


class ObsDistribution(BaseModel):
    agents_per_observation: List[Tuple[str, List[str]]] = Field(
        description="The list of observations with a list of agents that perceive the observation"
    )


class FormattedQuestion(BaseModel):
    agent_name: str = Field(
        description="The name of the agent that the question should be directed to"
    )
    question_observation: Observation = Field(
        description="The observation containing the question to be asked to the agent"
    )


class Simulation(BaseModel):
    agents: List[str] = Field(description="The list of agents in the simulation")
    agent_memories: Dict[str, List[str]] = Field(
        description="The list of memories for each agent"
    )
    question: str = Field(description="The question that was asked to the agent")
    reasoning: str = Field(
        description="The reasoning that the agent did to answer the question"
    )
    answer: str = Field(description="The answer that the agent gave to the question")


class ToMEngine:
    def __init__(
        self,
        agent_prompt: str = "",
        task_specific_instructions: str = "",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        existing_socialized_contexts: dict[str, Any] = {},
    ):
        """Initialize ToM engine.

        Args:
            example: Example scenario dictionary
            example_analysis: Example analysis dictionary
            model_name: Name of the model to use
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens for generation
            top_p: Top p parameter for generation
        """
        self.agent_prompt = agent_prompt
        self.task_specific_instructions = task_specific_instructions
        self.model_name = model_name
        self.temperature = temperature
        self.agents: Dict[str, LLMAgent] = {}
        self.current_time = 0
        self.existing_socialized_contexts = existing_socialized_contexts
        self.simulation: Simulation = Simulation(
            agents=[], agent_memories={}, question="", reasoning="", answer=""
        )

    def add_agent(self, name: str) -> None:
        self.agents[name] = LLMAgent(
            name, agent_prompt=self.agent_prompt, model_name=self.model_name
        )

    def reset_agents(self) -> None:
        self.agents = {}
        self.current_time = 0

    def set_task_specific_instructions(self, task_specific_instructions: str) -> None:
        self.task_specific_instructions = task_specific_instructions

    def set_agent_prompt(self, agent_prompt: str) -> None:
        self.agent_prompt = agent_prompt

    def get_simulation(self) -> Simulation:
        self.simulation.agents = list(self.agents.keys())
        self.simulation.agent_memories = {
            agent: [message.last_turn for message in self.agents[agent].message_history]
            for agent in self.agents
        }
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
        assert isinstance(
            agent_memory, ObsDistribution
        ), "Output parser did not return an ObsDistribution"
        return agent_memory.agents_per_observation

    async def initialize_simulation(
        self,
        agent_names: list[str],
        observations_with_perceivers: List[Tuple[str, List[str]]],
    ) -> None:
        for agent_name in agent_names:
            self.add_agent(agent_name)
        for obs, perceivers in observations_with_perceivers:
            for agent in perceivers:
                self.agents[agent].message_history.append(
                    Observation(
                        agent_name=agent,
                        last_turn=obs,
                        turn_number=self.current_time,
                        available_actions=[
                            "none",
                            "speak",
                            "non-verbal communication",
                            "action",
                            "leave",
                        ],
                    )
                )

    async def initialize_simulation_from_socialized_context(
        self, socialized_context: dict[str, Any]
    ) -> None:
        socialized_events = socialized_context["socialized_context"]
        agent_names = socialized_context["agents_names"]
        last_action = ""
        for agent_name in agent_names:
            self.add_agent(agent_name)
        for step in socialized_events:
            if step['state'] == '[SAME AS LAST ACTION]':
                step['state'] = last_action
            for agent_name, observation in step["observations"].items():
                if observation == "none":
                    continue
                if observation == "[SAME AS STATE]":
                    observation = step["state"]
                # avoid adding the same action twice
                if self.agents[agent_name].message_history and observation in self.agents[agent_name].message_history[-1].last_turn:
                    continue
                self.agents[agent_name].message_history.append(
                    Observation(
                        agent_name=agent_name,
                        last_turn=observation,
                        turn_number=self.current_time,
                        available_actions=[
                            "none",
                            "speak",
                            "non-verbal communication",
                            "action",
                            "leave",
                        ],
                    )
                )
            for agent_name, action in step["actions"].items():
                if action == "none":
                    continue
                if not action.startswith("agent_name"):
                    action = f"{agent_name} {action}"
                last_action = action
                self.agents[agent_name].message_history.append(
                    Observation(
                        agent_name=agent_name,
                        last_turn=action,
                        turn_number=self.current_time,
                        available_actions=[
                            "none",
                            "speak",
                            "non-verbal communication",
                            "action",
                            "leave",
                        ],
                    )
                )

    async def reason_about_belief(
        self,
        question: str,
        agents: list[str],
        target_agent: str | None = None,
        answer_candidates: list[str] | None = None,
    ) -> Tuple[str, str]:
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
                available_actions=[
                    "none",
                    "speak",
                    "non-verbal communication",
                    "action",
                    "leave",
                ],
            )
            if answer_candidates:
                question_observation.last_turn += f"The question should be answered by one of the following candidates: {answer_candidates}"
        assert target_agent in self.agents, f"Agent {target_agent} not found in agents"
        action = await self.agents[target_agent].aact(question_observation)
        assert (
            action is not None
        ), f"Action is None for {question_observation.last_turn}"
        reasoning_and_answer = action.argument
        try:
            reasoning = reasoning_and_answer.split("<reasoning>")[1].split(
                "</reasoning>"
            )[0]
            answer = reasoning_and_answer.split("<answer>")[1].split("</answer>")[0]
        except Exception:
            reasoning = ""
            answer = reasoning_and_answer
        self.simulation.reasoning = reasoning
        self.simulation.answer = answer
        self.simulation.question = question
        return reasoning, answer

    async def socialize_context(
        self, context: str, example_analysis: str = "", feedback: str = ""
    ) -> SocializedContext:
        """
        Analyzes and socializes context for the simulation.

        Args:
            context: A string describing the social context/situation
            example_analysis: Example analysis to guide the socialization
            use_critic: Whether to use the critic feedback loop
            example_patterns: Example error patterns to check against in the critic
            max_attempts: Maximum number of attempts to improve the context

        Returns:
            SocializedContext object containing the analyzed context
        """
        template = (
            "Please analyze the following narrative.\n\n" "Context: {context}\n\n"
        )

        input_values = {"context": context}

        if self.task_specific_instructions:
            template += "Task specific instructions: {task_specific_instructions}\n\n"
            input_values["task_specific_instructions"] = self.task_specific_instructions

        if example_analysis:
            template += "Example analysis: {example_analysis}\n\n"
            input_values["example_analysis"] = example_analysis

        if feedback:
            template += "Previous attempt had these issues. Please fix them based on the previous attempt and feedback below:\n{feedback}\n\n"
            input_values["feedback"] = feedback

        template += "Follow these format instructions:\n{format_instructions}"
        socialized_context = await agenerate(
            model_name=self.model_name,
            template=template,
            input_values=input_values,
            temperature=self.temperature,
            output_parser=PydanticOutputParser(pydantic_object=SocializedContext),
            structured_output=True,
        )
        # save the socialized context to a file
        with open("socialized_context.json", "w") as f:
            json.dump(socialized_context.model_dump(), f, indent=2)
        assert isinstance(
            socialized_context, SocializedContext
        ), "Socialized context is not a SocializedContext"
        return socialized_context

    async def critique_and_improve_context(
        self,
        socialized_context: SocializedContext,
        context: str,
        example_analysis: str = "",
        example_patterns: str = "",
        max_attempts: int = 1,
    ) -> SocializedContext:
        """
        Critiques and improves a socialized context through iterative feedback.

        Args:
            socialized_context: Initial socialized context to improve
            context: Original context string
            example_analysis: Example analysis to guide the socialization
            example_patterns: Example error patterns to check against
            max_attempts: Maximum number of improvement attempts

        Returns:
            Improved SocializedContext object
        """
        attempts = 0
        current_context = socialized_context
        critique = ""

        while attempts < max_attempts:
            # Convert to JSON string for the critic
            context_json = json.dumps(current_context.model_dump(), indent=2)
            # Get critic feedback
            is_good, critique = await self.critic_socialize_context(
                context_json, example_patterns
            )
            if is_good:
                print(f"Socialized context is good after {attempts} critique(s).")
                return current_context

            print(f"Attempt {attempts + 1} feedback: {critique}")

            # Generate new context without triggering the critic again to avoid infinite recursion
            current_context = await self.socialize_context(
                context=context, example_analysis=example_analysis, feedback=critique
            )

            attempts += 1

        print(
            f"Reached maximum attempts ({max_attempts}). Returning the last context version."
        )
        return current_context

    async def critic_socialize_context(
        self, socialized_context: str, example_patterns: str
    ) -> Tuple[bool, str]:
        """
        Evaluates whether the socialized context is good enough based on provided patterns.

        Args:
            socialized_context: A JSON string containing the socialized context
            example_patterns: Patterns or criteria to evaluate the context against

        Returns:
            String containing "yes" if the context is good, or specific feedback if not
        """
        # Get format instructions for SocializedContext
        format_instructions = PydanticOutputParser(
            pydantic_object=SocializedContext
        ).get_format_instructions()

        template = (
            "You are a critical evaluator of socialized context for simulations.\n\n"
            "Please evaluate the following socialized context:\n{socialized_context}\n\n"
            "Task specific requirements and example errors patterns/criteria for bad socialized context:\n{example_patterns}\n\n"
            "The expected format for a good SocializedContext is:\n{format_instructions}\n\n"
            "Evaluate if this socialized context is good enough for simulation. Consider:\n"
            "If the context is good enough, respond with 'yes'.\n"
            "If the context is not good enough, provide specific feedback on what needs to be improved."
        )

        evaluation = await agenerate(
            model_name=self.model_name,
            template=template,
            input_values={
                "socialized_context": socialized_context,
                "example_patterns": example_patterns,
                "format_instructions": format_instructions,
            },
            temperature=0.3,
            output_parser=StrOutputParser(),
        )
        if evaluation.startswith("yes"):
            return True, evaluation
        else:
            return (
                False,
                f"Previous attempt: {socialized_context}\n\nFeedback: {evaluation}",
            )
