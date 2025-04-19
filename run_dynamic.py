from sotopia.cli import benchmark
from social_world_model.agents import SocialWorldModelAgent 
from sotopia.agents import LLMAgent

if __name__ == "__main__":
    benchmark(
        models=[
            "gpt-4o-2024-08-06",
        ],
        partner_model="gpt-4o-2024-08-06",
        agent_class=SocialWorldModelAgent,
        tag="social_world_model_trial_17_social_world_model_agent",
        batch_size=100,
        push_to_db=True,
        evaluator_model="o3-2025-04-16",
        task="normal",
    )
