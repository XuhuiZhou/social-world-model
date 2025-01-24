from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from social_world_model.database import Observation, AgentAction

@dataclass
class AgentDependencies:
    pass

class AgentResult(BaseModel):
    response: str = Field(description='Response from the agent')

pydantic_agent = Agent(
    'openai:gpt-4o',
    deps_type=AgentDependencies,
    result_type=AgentResult,
    system_prompt='You are an AI agent. Provide concise and relevant responses.'
)

@pydantic_agent.system_prompt
async def add_context(ctx: RunContext[AgentDependencies]) -> str:
    return "This is additional context for the agent."

@pydantic_agent.tool
async def example_tool(ctx: RunContext[AgentDependencies]) -> str:
    """Example tool function."""
    return "Tool response"

class PydanticAgent:
    def __init__(
        self,
        agent_name: str,
        input_channels: list[str] = ['observation'],
        output_channel: str = "action",
        query_interval: int = 0,
        node_name: str = "pydantic_agent",
        agent_prompt: str = "",
        model_name: str = "gpt-4o",
        redis_url: str = "redis://localhost:6379/0",
    ):
        self.name = agent_name
        self.input_channels = input_channels
        self.output_channel = output_channel
        self.query_interval = query_interval
        self.node_name = node_name
        self.agent_prompt = agent_prompt
        self.model_name = model_name
        self.redis_url = redis_url
        self.count_ticks = 0
        self.message_history: list[Observation] = []

    def _format_message_history(self, message_history: list[Observation]) -> str:
        return "\n".join(message.to_natural_language() for message in message_history)

    async def aact(self, obs: Observation) -> AgentAction:
        if obs.turn_number == -1:
            return AgentAction(
                agent_name=self.name,
                output_channel=self.output_channel,
                action_type="none",
                argument=self.model_name,
            )

        self.message_history.append(obs)

        history = self._format_message_history(self.message_history)
        result = await pydantic_agent.run(history, deps=AgentDependencies())
        action = result.data['response']

        return AgentAction(
            agent_name=self.name,
            output_channel=self.output_channel,
            action_type="speak",
            argument=action,
        )

