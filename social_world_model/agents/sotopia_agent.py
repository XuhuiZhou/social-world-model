from sotopia.agents import LLMAgent
from sotopia.database import AgentProfile
from sotopia.messages import AgentAction, Observation
from sotopia.generation_utils import agenerate_action
from social_world_model.database import SocializedContext, SocializedStructure
from social_world_model.social_world_model import SocialWorldModel
from typing import Optional
import logging

class SocialWorldModelAgent(LLMAgent):
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
        model_name: str = "gpt-4o-mini",
        script_like: bool = False,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
            model_name=model_name,
            script_like=script_like,
        )
        self.socialized_context = SocializedContext(
            **{
                "agents_names": [self.agent_name, "X"],
                "socialized_context": [],
                "task_specific_instructions": "<same_as_next_state /> means the content is the same as the state of the next timestep.",
            }
        )
        self.engine = SocialWorldModel(model_name=self.model_name)
        self.additional_instructions = f"Please additionally add the <mental_state> </mental_state> of each agent in their observations reacting to {self.agent_name}'s action. More specifically, fill in the social goal of the agents in the <mental_state> </mental_state>."
        self.last_socialized_context_step: Optional[SocializedStructure] = None
    
    async def predict_socialized_context(self, obs: Observation, action: AgentAction) -> SocializedContext:
        """
        Updates the socialized context based on the current observation and action.
        
        Args:
            obs: The current observation
            action: The current action taken by the agent
            
        Returns:
            Updated socialized context
        """
        if self.socialized_context.socialized_context == []:
            first_obs_content = obs.last_turn
            try:
                extracted_names = first_obs_content.split("Participants: ")[1].split("\n")[0].split("and ")
                extracted_names = [x.strip() for x in extracted_names]
                assert self.agent_name in extracted_names
                self.socialized_context.agents_names = [extracted_names[0], extracted_names[1]]
                self.partner_name = extracted_names[1-extracted_names.index(self.agent_name)]
            except Exception as e:
                pass
            
            current_step = SocializedStructure(
                timestep=str(obs.turn_number),
                state="none",
                observations={self.agent_name: obs.to_natural_language(), self.partner_name: "<unknown />"},
                actions={self.agent_name: "none", self.partner_name: "<same_as_next_state />"} if action.action_type == "none" else {self.agent_name: action.to_natural_language(), self.partner_name: "none"}
            )
        else:
            current_step = SocializedStructure(
                timestep=str(obs.turn_number),
                state=obs.to_natural_language(),
                observations={self.agent_name: "<same_as_state />", self.partner_name: "<same_as_state />"},
                actions={self.agent_name: "none", self.partner_name: "<same_as_next_state />"} if action.action_type == "none" else {self.agent_name: action.to_natural_language(), self.partner_name: "none"}
            )

        self.socialized_context.socialized_context.append(current_step)
        if action.action_type != "none":
            socialized_context = await self.engine.simulate_one_step(self.socialized_context, self.additional_instructions)
            self.socialized_context = socialized_context
        return self.socialized_context
    
    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)
        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            none_action = AgentAction(action_type="none", argument="")
            await self.predict_socialized_context(obs, none_action)
            return none_action
        else:
            action = await agenerate_action(
                self.model_name,
                history="\n".join(f"{y.to_natural_language()}" for x, y in self.inbox),
                turn_number=obs.turn_number,
                action_types=obs.available_actions,
                agent=self.agent_name,
                goal=self.goal,
                script_like=self.script_like,
                structured_output=True,
            )
            assert isinstance(action, AgentAction)
            try:
                next_socialized_context = await self.predict_socialized_context(obs, action)
                self.last_socialized_context_step = next_socialized_context.socialized_context[-1]
            except Exception as e:
                logging.warning(f"Error predicting socialized context: {e}")
                return action
            # Regenerate the action based on the imagined last socialized context step
            socialized_context_info = f"""Imagined context: you might think of the following action:
{action.to_natural_language()}
And here's the potential mental states and reactions of the other agent:
- Other agent's mental state: {self.last_socialized_context_step.observations[self.partner_name]}
- Expected other agent's reaction: {self.last_socialized_context_step.actions[self.partner_name]}
Based on the imagined context, refine your action if needed.
"""
            refined_action = await agenerate_action(
                self.model_name,
                history=f"\n".join(f"{y.to_natural_language()}" for x, y in self.inbox) + "\n" + socialized_context_info,
                turn_number=obs.turn_number,
                action_types=obs.available_actions,
                agent=self.agent_name,
                goal=self.goal,
                script_like=self.script_like,
                structured_output=True,
            )
            assert isinstance(refined_action, AgentAction)
            return refined_action
