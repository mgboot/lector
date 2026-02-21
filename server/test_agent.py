"""Smoke-test: verify that the agent can be created and respond to a message."""

import asyncio

from agent import create_agent


async def main():
    print("Creating agent…")
    try:
        agent = create_agent()
    except Exception as e:
        print(f"FAILED to create agent: {e}")
        return

    print("Agent created. Sending test message…")
    try:
        session = agent.create_session()
        result = await agent.run("Translate 'hello' to Latin.", session=session)
        print(f"Response: {result}")
    except Exception as e:
        print(f"FAILED during agent.run(): {e}")
        return

    print("\nSmoke test passed!")


if __name__ == "__main__":
    asyncio.run(main())
