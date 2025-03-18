from rich import print
from social_world_model.social_world_model import SocialWorldModel


async def run_sally_anne_test() -> None:
    engine = SocialWorldModel()
    # Run the full scenario
    scenario = (
        "Sally puts a marble in her basket, leaves the room, "
        "and while she's gone, Anne moves the marble to her own box."
    )

    questions = [
        "Where Sally believes the marble is",
        # 'Where Anne believes the marble is',
        # 'Where Sally believes Anne believes the marble is',
        # 'Where Anne believes Sally believes the marble is',
    ]

    await engine.distribute_observations(scenario)

    for question in questions:
        answer = await engine.reason_about_belief(question, list(engine.agents.keys()))
        print(f"{question}: {answer}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_sally_anne_test())


print("Hello")
