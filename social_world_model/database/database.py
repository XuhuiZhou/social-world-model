from pydantic import Field, BaseModel
from sotopia.messages import ActionType


class SocializedStructureForModel(BaseModel):
    timestep: str = Field(
        description="The timestep of the current socialized structure, it could be a integer number or a description of the time of the state.",
    )
    state: str = Field(
        description="The current state of the world (including all the agents) at this timestep. Important note: this is the state before the action is taken (e.g., the initial state could be 'none' at the beginning if there are no prior contexts before the interaction starts)."
    )
    observations: list[str] = Field(
        description="The observations for each agent in the social world at this timestep (similar to the definition in partial observable Markov Decision Process, observation is derived from the obervation function with the current state as the argument). Note that the different agents may have different observations. The observation would go into corresponding agent's memory, so make sure the observation is clear for the agent to understand (first person perspective narrative is preferred). If it is the same as the current state, use the special tag '<same_as_state />' to indicate the observation. For the internal thoughts, beliefs, or emotions of the agent that is not directly observable by other agents, use the special tag '<mental_state>...</mental_state>' to indicate the internal observation. Put 'none' if the agent does not observe anything at this timestep. Important note: this is the observation before the action is taken (e.g., the observation could be 'none' at the beginning if there are no prior contexts before the interaction starts). The format for each entry in the list is: 'agent_name: observation'"
    )
    actions: list[str] = Field(
        description="The actions for each agent in the social world at this timestep. The length of the list should be the same as the number of agents. Put 'none' if the agent does not take any action at this timestep. The format for each entry in the list is: 'agent_name: action'"
    )


class SocializedStructure(BaseModel):
    timestep: str
    state: str
    observations: dict[str, str]
    actions: dict[str, str]

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
    agents_names: list[str]
    socialized_context: list[SocializedStructure]
    context_manual: str = Field(
        description="The manual of how to interpret the socialized context, it should come from the prompt of generating the socialized context."
    )

    def to_natural_language(self) -> str:
        return "### Socialized Context\n" + self.model_dump_json(indent=2)


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
        description="whether to speak at this turn or choose to not do anything"
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
