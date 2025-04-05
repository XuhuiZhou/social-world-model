from typing import Dict, List, Tuple
from sotopia.generation_utils import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from social_world_model.agents import LLMAgent
from social_world_model.database import (
    Observation,
    SocializedContext,
    SocializedContextForModel,
    SocializedStructure,
)
from sotopia.generation_utils import agenerate
from social_world_model.engine import dictlize, GENERAL_GUIDELINES
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


class SocialWorldModel:
    def __init__(
        self,
        agent_prompt: str = "",
        task_specific_instructions: str = "",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        existing_socialized_contexts: dict[str, SocializedContext] = {},
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

    async def decode_socialized_context(
        self, socialized_context: SocializedContext
    ) -> SocializedContext:
        """
        Decode special symbols in the socialized context.

        Args:
            socialized_context: The socialized context with potential special symbols

        Returns:
            Decoded socialized context with special symbols replaced with actual text
        """
        last_action = ""
        last_action_agent = ""
        # Create a new SocializedContext object with the same attributes
        decoded_context = SocializedContext(
            agents_names=socialized_context.agents_names.copy(),
            socialized_context=[],
            context_manual=socialized_context.context_manual,
        )

        # Process each socialized structure
        for step in socialized_context.socialized_context:
            # Create a new structure for the current step
            decoded_step = SocializedStructure(
                timestep=step.timestep,
                state=step.state.replace("<same_as_last_action />", last_action),
                observations={},
                actions={},
            )

            # Process observations
            for agent_name, observation in step.observations.items():
                decoded_step.observations[agent_name] = observation.replace(
                    "<same_as_state />", decoded_step.state
                )
                if observation != last_action and agent_name == last_action_agent:
                    if observation == "none":
                        decoded_step.observations[agent_name] = last_action
                    else:
                        decoded_step.observations[agent_name] = (
                            last_action + observation
                        )

            # Process actions
            for agent_name, action in step.actions.items():
                if action != "none" and not action.startswith("agent_name"):
                    action = f"{agent_name} {action}"
                    # Update last_action for future reference
                    # TODO: This is a strong assumption that only one action is taken per timestep
                    last_action = action
                    last_action_agent = agent_name

                decoded_step.actions[agent_name] = action

            # Add the decoded step to the new context
            decoded_context.socialized_context.append(decoded_step)

        return decoded_context

    async def initialize_simulation_from_socialized_context(
        self, socialized_context: SocializedContext
    ) -> None:
        """
        Initialize simulation from a socialized context.

        Args:
            socialized_context: The socialized context to initialize from
        """
        decoded_socialized_context = await self.decode_socialized_context(
            socialized_context
        )
        # Use the SocializedContext object directly
        socialized_events = decoded_socialized_context.socialized_context
        agent_names = decoded_socialized_context.agents_names

        # Add agents
        for agent_name in agent_names:
            self.add_agent(agent_name)

        # Process events
        for step in socialized_events:
            # Add observations to agent message history
            for agent_name, observation in step.observations.items():
                if observation == "none":
                    continue
                self.agents[agent_name].message_history.append(
                    Observation(
                        agent_name=agent_name,
                        last_turn=observation,
                        turn_number=step.timestep,
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
        self, context: str, example_analysis: str = "", feedback: str = "", critic_and_improve: bool = False, template: str = ""
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
        if not template:
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
            output_parser=PydanticOutputParser(
                pydantic_object=SocializedContextForModel
            ),
            structured_output=True,
        )
        assert isinstance(
            socialized_context, SocializedContextForModel
        ), "Socialized context is not a SocializedContext"
        socialized_context_dict = dictlize(socialized_context)
        context_manual = f"Here's how to interpret the socialized context: \n{json.dumps(SocializedContextForModel.model_json_schema(), indent=2)}\nTask specific instructions:{self.task_specific_instructions}"
        socialized_context_processed = SocializedContext(
            **socialized_context_dict, context_manual=context_manual
        )
        if critic_and_improve:
            socialized_context_processed = await self.critique_and_improve_context(
                socialized_context=socialized_context_processed,
                context=context,
                max_attempts=3,
            )
        return socialized_context_processed

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
        self, socialized_context: str, example_patterns: str = ""
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
            "Here are some general guidelines for good socialized context:\n{GENERAL_GUIDELINES}\n\n"
            "Task specific requirements and example errors patterns/criteria for bad socialized context:\n{example_patterns}\n\n"
            "The expected format for a good SocializedContext is:\n{format_instructions}\n\n"
            "Evaluate if this socialized context is good enough for simulation. Consider:\n"
            "If the context is good enough, respond with 'yes', following with the reasoning of the judgement.\n"
            "If the context is not good enough, respond with 'no', following with the reasoning of the judgement and provide specific feedback on what needs to be improved."
        )

        evaluation = await agenerate(
            model_name=self.model_name,
            template=template,
            input_values={
                "socialized_context": socialized_context,
                "GENERAL_GUIDELINES": GENERAL_GUIDELINES,
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
    
    async def simulate_socialized_context(self, context: str, socialized_context: SocializedContext|None = None) -> SocializedContext:
        """
        Simulates the socialized context from the given context.

        Args:
            context: The original context to simulate from
            socialized_context: The socialized context to simulate from
        """
        if socialized_context:
            processed_context = f"### Original Context:\n{context}\n\n{socialized_context.to_natural_language()}\n\n"
        else:
            processed_context = f"### Original Context:\n{context}\n\n"
        template = (
            "Please analyze the current context and roll out the future socialized context from the following context (please be creative about the future roll out):\n{context}\n\n"
        )
        socialized_context = await self.socialize_context(context=processed_context, critic_and_improve=False, template=template)
        return socialized_context