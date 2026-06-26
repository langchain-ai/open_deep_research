# odr_min.py
import os
import argparse
import json
import asyncio
import uuid
import random
from dotenv import load_dotenv
from tqdm import tqdm
from langgraph.checkpoint.memory import MemorySaver
from open_deep_research.deep_researcher import deep_researcher_builder

# def extract_report(state: dict) -> str:
#     # Prefer markdown fields; fall back to assistant message
#     for k in ("final_report_md", "report_markdown", "report_md", "final_report"):
#         v = state.get(k)
#         if isinstance(v, str) and v.strip():
#             return v
#     msgs = state.get("messages")
#     if isinstance(msgs, list):
#         for m in reversed(msgs):
#             if m.get("role") == "assistant" and m.get("content"):
#                 return m["content"]
#     return json.dumps(state, ensure_ascii=False, indent=2)

def extract_report(state: dict) -> str:
    final_report = state.get("final_report")
    return final_report

async def arun(data: dict, output_dir: str):
    
    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())

    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "max_structured_output_retries": 3,
            "allow_clarification": False,
            "max_concurrent_research_units": 10,
            "search_api": "serper",
            "max_researcher_iterations": 3,
            "max_react_tool_calls": 10,


            # "summarization_model": "azure_openai:gpt-4o-mini",
            # "summarization_model_max_tokens": 8192,
            # "research_model": "azure_openai:gpt-4o-mini",
            # "research_model_max_tokens": 10000,
            # "compression_model": "azure_openai:gpt-4o-mini",
            # "compression_model_max_tokens": 10000,
            # "final_report_model": "azure_openai:gpt-4o-mini",
            # "final_report_model_max_tokens": 10000,
            "summarization_model": "openai:gpt-4.1-mini",
            "summarization_model_max_tokens": 8192,
            "research_model": "openai:gpt-4.1",
            "research_model_max_tokens": 10000,
            "compression_model": "openai:gpt-4.1",
            "compression_model_max_tokens": 10000,
            "final_report_model": "openai:gpt-4.1",
            "final_report_model_max_tokens": 10000,
        }
    }

    prompt = data['prompt']
    state = await graph.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config
    )

    os.makedirs(output_dir, exist_ok=True)
    report = extract_report(state)
    out_path = os.path.join(output_dir, f"{data['id']}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved to file: {out_path}")

def read_jsonl(file_path: str):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def parse_args():
    parser = argparse.ArgumentParser(description="Deep Research (ODR)")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to prompts.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save reports")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()

def main():
    load_dotenv("../../.env")
    args = parse_args()

    prompts_data = read_jsonl(args.prompt_file)

    if args.debug:
        prompts_data = random.sample(prompts_data, min(4, len(prompts_data)))

    print(f"Starting Deep Research for {len(prompts_data)} prompts")
    for  prompt in tqdm(prompts_data, desc="Running ODR"):
        asyncio.run(arun(prompt, args.output_dir))

if __name__ == "__main__":
    main()
