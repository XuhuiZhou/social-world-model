import pytest
from social_world_model.social_world_model import SocialWorldModel
from social_world_model.database import SocializedContext, SocializedStructure

@pytest.mark.asyncio
async def test_simulate_one_step() -> None:
    # Initialize the model
    model = SocialWorldModel(model_name="o3-mini-2025-01-31")
    
    # Create a simple socialized context with one step
    initial_context_dict = {
        "agents_names": ["Alice", "Bob"],
        "socialized_context": [
            {
                "timestep": "1",
                "state": "Alice and Bob are in a room",
                "observations": {
                    "Alice": "I see Bob in the room",
                    "Bob": "I see Alice in the room"
                },
                "actions": {
                    "Alice": "none",
                    "Bob": "none"
                }
            }
        ],
        "context_manual": "Test context"
    }
    # Simulate one step
    new_context = await model.simulate_one_step(SocializedContext(**initial_context_dict))
    print(new_context)
    
    # Verify the new context
    assert isinstance(new_context, SocializedContext)
    assert len(new_context.socialized_context) == 2  # Should have one more step
    assert new_context.agents_names == ["Alice", "Bob"]  # Agents should be preserved
    assert new_context.context_manual == "Test context"  # Manual should be preserved
    
    # Verify the new step has the expected structure
    new_step = new_context.socialized_context[-1]
    assert isinstance(new_step, SocializedStructure)
    assert "timestep" in new_step.__dict__
    assert "state" in new_step.__dict__
    assert "observations" in new_step.__dict__
    assert "actions" in new_step.__dict__
    
    # Verify the new step has observations and actions for both agents
    assert "Alice" in new_step.observations
    assert "Bob" in new_step.observations
    assert "Alice" in new_step.actions
    assert "Bob" in new_step.actions 