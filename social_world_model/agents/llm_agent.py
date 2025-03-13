from social_world_model.database import Observation, AgentAction
from sotopia.generation_utils import agenerate, StrOutputParser


class LLMAgent:
    def __init__(
        self,
        agent_name: str,
        input_channels: list[str] = ["observation"],
        output_channel: str = "action",
        query_interval: int = 0,
        node_name: str = "llm_agent",
        agent_prompt: str = "",
        model_name: str = "gpt-4o",
        redis_url: str = "redis://localhost:6379/0",
    ):
        # super().__init__(
        #     [(input_channel, Observation) for input_channel in input_channels],
        #     [(output_channel, AgentAction)],
        #     redis_url,
        #     node_name,
        # )
        self.output_channel = output_channel
        self.query_interval = query_interval
        self.count_ticks = 0
        self.message_history: list[Observation] = []
        self.name = agent_name
        self.model_name = model_name
        self.agent_prompt = agent_prompt + " " if agent_prompt else ""

    def _format_message_history(self, message_history: list[Observation]) -> str:
        ## TODO: akhatua Fix the mapping of action to be gramatically correct
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
        action: str = await agenerate(
            model_name=self.model_name,
            template="Imagine that you are {agent_name} in the scenario. {agent_prompt}Below is the previous history of the interaction.\n"
            "{message_history}\n",
            input_values={
                "message_history": history,
                "agent_name": self.name,
                "agent_prompt": self.agent_prompt,
            },
            temperature=0.7,
            output_parser=StrOutputParser(),
        )

        return AgentAction(
            agent_name=self.name,
            output_channel=self.output_channel,
            action_type="speak",
            argument=action,
        )
