import sys
import os
import asyncio
import argparse

# Add the src directory to the Python path before other imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from langchain_core.messages import HumanMessage
from open_deep_research.deep_researcher import deep_researcher

async def main():
    parser = argparse.ArgumentParser(description="Run the deep research agent.")
    parser.add_argument("query", type=str, help="The research query.")
    args = parser.parse_args()

    config = {"configurable": {}}
    inputs = {"messages": [HumanMessage(content=args.query)]}
    final_report = ""

    async for event in deep_researcher.astream_events(inputs, config=config, version="v2"):
        kind = event["event"]
        if kind == "on_chain_end":
            if event["name"] == "final_report_generation":
                final_report = event["data"]["output"]["final_report"]
                print("Final report generated.")
                break

    if final_report:
        with open("final_report.md", "w") as f:
            f.write(final_report)
        print("Final report saved to final_report.md")
    else:
        print("Could not generate final report.")

if __name__ == "__main__":
    asyncio.run(main())
