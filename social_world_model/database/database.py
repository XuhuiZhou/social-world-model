from pydantic import Field, BaseModel
from sotopia.messages import ActionType
from typing import Any
import json


class SocializedStructureForModel(BaseModel):
    timestep: str = Field(
        description="The timestep of the current socialized structure, it could be a integer number or a description of the time of the state.",
    )
    state: str = Field(
        description="The current state of the world (including all the agents) at this timestep. Important note: this is the state before the action is taken (e.g., the initial state could be 'none' at the beginning if there are no prior contexts before the interaction starts)."
    )
    observations: list[str] = Field(
        description="The observations for each agent in the social world at this timestep. Note that the different agents may have different observations. The observation would go into corresponding agent's memory, so make sure the observation is clear for the agent to understand (first person perspective narrative is preferred). 1. If the observation covers the current state, use the special tag '<same_as_state />' to indicate that. 2. If the observation covers last timestep agents' actions, use '<same_as_last_action_x />' to cover that, x means the index of the agents. 3. For the internal thoughts, beliefs, or emotions of the agent that is not directly observable by other agents, use the special tag '<mental_state>...</mental_state>' to indicate the internal observation. You can of course combine these tags and add extra information after the tags (seperated by space). 4. Put 'none' if the agent does not observe anything at this timestep. Important note: this is the observation before the action is taken (e.g., the observation could be 'none' at the beginning if there are no prior contexts before the interaction starts). The format for each entry in the list is: 'agent_name: observation'"
    )
    actions: list[str] = Field(
        description="The actions for each agent in the social world at this timestep. The length of the list should be the same as the number of agents. Put 'none' if the agent does not take any action at this timestep. The format for each entry in the list is: 'agent_name: action'"
    )


class SocializedStructure(BaseModel):
    timestep: str = Field(
        description="The timestep of the current socialized structure, it could be a integer number or a description of the time of the state."
    )
    state: str = Field(
        description="The current state of the world (including all the agents) at this timestep. Important note: this is the state before the action is taken (e.g., the initial state could be 'none' at the beginning if there are no prior contexts before the interaction starts)."
    )
    observations: dict[str, str] = Field(
        description="The observations for each agent in the social world at this timestep. Note that the different agents may have different observations.  1. The special tag '<same_as_state />' indicates the observation covers the current state. 2. The special tag '<same_as_last_action_x />' indicates the observation covers the last timestep agents' actions, x means the index of the agents. If no x provided, it means the observation covers the last timestep agents' actions. 3. The special tag '<mental_state>...</mental_state>' indicates the mental state of the agent. 4. 'none' means the agent does not observe anything at this timestep. Important note: this is the observation before the action is taken (e.g., the observation could be 'none' at the beginning if there are no prior contexts before the interaction starts)."
    )
    actions: dict[str, str] = Field(
        description="The actions for each agent in the social world at this timestep. 'none' represents that the agent does not take any action at this timestep."
    )

    def to_natural_language(self, timestep: int) -> str:
        return (
            "At timestep "
            + str(timestep)
            + ":\nState: "
            + self.state
            + "\nObservations: "
            + "\n".join([f"{key}: {value}" for key, value in self.observations.items()])
            + "\nActions: "
            + "\n".join([f"{key}: {value}" for key, value in self.actions.items()])
        )


class SocializedContextForModel(BaseModel):
    agents_names: list[str] = Field(description="The names of the agents")
    socialized_context: list[SocializedStructureForModel] = Field(
        description="A list of SocializedStructureForModel objects, each representing a timestep of the social world. At the last timestep, all agents' actions should be 'none' as they have already completed the interaction."
    )


class SocializedContext(BaseModel):
    agents_names: list[str] = Field(description="The names of the agents")
    socialized_context: list[SocializedStructure] = Field(
        description="A list of SocializedStructure objects, each representing a timestep of the social world."
    )
    context_manual: str = Field(
        description="The manual of how to interpret the socialized context, it should come from the prompt of generating the socialized context."
    )

    def __init__(self, **data: Any) -> None:
        # Add timesteps if not present
        if "socialized_context" in data:
            for i, structure in enumerate(data["socialized_context"]):
                if "timestep" not in structure:
                    structure["timestep"] = str(i)

        # Add context manual if not present
        if "context_manual" not in data:
            data["context_manual"] = self.create_context_manual(
                data.get(
                    "task_specific_instructions",
                    "no domain specific instructions when generating the socialized context",
                )
            )

        super().__init__(**data)

    def to_natural_language(self) -> str:
        context_manual = self.context_manual

        return (
            "### Socialized Context (the analysis of the original context)\n"
            + self.model_dump_json(indent=2, exclude={"context_manual"})
            + "\n\n"
            + context_manual
        )

    def create_context_manual(self, task_specific_instructions: str) -> str:
        return f"#### Context Manual\nHere's how to interpret the above socialized context (i.e., the json schema): \n{json.dumps(SocializedContext.model_json_schema(), indent=2)}\n#### Here's the domain specific instuctions when generating the socialized context (should help you better understand the socialized context):\n{task_specific_instructions}"


class SocialSimulation(BaseModel):
    simulations: list[SocializedContext] = Field(
        description="A list of SocializedContext objects, each representing a simulation of the social world."
    )

    def to_natural_language(self) -> str:
        context_manual = (
            self.simulations[0].context_manual
        )  # TODO: Assume all the simulations have the same context manual
        return (
            "### Social Simulation (the simulations based on the original context)\n\n"
            + "\n\n".join(
                [
                    f"#### Possible Social Simulation {index}:\n"
                    + simulation.model_dump_json(indent=2, exclude={"context_manual"})
                    for index, simulation in enumerate(self.simulations)
                ]
            )
            + "\n\n"
            + context_manual
        )


class Observation(BaseModel):
    agent_name: str = Field(description="the name of the agent")
    last_turn: str = Field(description="the last turn of the conversation")
    turn_number: int = Field(description="the turn number of the conversation")
    available_actions: list[ActionType] = Field(description="the available actions")

    def to_natural_language(self) -> str:
        return f"{self.last_turn}\n"


class AgentAction(BaseModel):
    agent_name: str = Field(description="the name of the agent")
    output_channel: str = Field(description="the name of the output channel")
    action_type: ActionType = Field(
        description="whether to speak at this turn or choose to not do anything",
        pattern="^(none|speak|non-verbal communication|action|leave)$"
    )
    argument: str = Field(
        description="the utterance if choose to speak, the expression or gesture if choose non-verbal communication, or the physical action if choose action"
    )

    def to_natural_language(self) -> str:
        match self.action_type:
            case "none":
                return "did nothing"
            case "speak":
                return f'said: "{self.argument}"'
            case "non-verbal communication":
                return f"[{self.action_type}] {self.argument}"
            case "action":
                return f"[{self.action_type}] {self.argument}"
            case "leave":
                return "left the conversation"
