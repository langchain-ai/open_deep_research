import argparse
import json
import sys
import urllib.parse
from typing import Any, Dict, List, Optional

import httpx


def build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []
    if args.message:
        messages.append({"type": "human", "content": args.message})
    elif args.input_file:
        content = open(args.input_file, "r", encoding="utf-8").read()
        messages.append({"type": "human", "content": content})
    else:
        raise SystemExit("Provide --message or --input-file")

    configurable: Dict[str, Any] = {}
    if args.thread_id:
        configurable["thread_id"] = args.thread_id

    # Model/search overrides
    if args.research_model:
        configurable["research_model"] = args.research_model
    if args.final_model:
        configurable["final_report_model"] = args.final_model
    if args.summarization_model:
        configurable["summarization_model"] = args.summarization_model
    if args.compression_model:
        configurable["compression_model"] = args.compression_model
    if args.search_api:
        configurable["search_api"] = args.search_api

    # Agenticity controls
    if args.max_researcher_iterations is not None:
        configurable["max_researcher_iterations"] = args.max_researcher_iterations
    if args.max_react_tool_calls is not None:
        configurable["max_react_tool_calls"] = args.max_react_tool_calls
    if args.max_concurrent_research_units is not None:
        configurable["max_concurrent_research_units"] = args.max_concurrent_research_units

    # Optional API keys via config (requires server env GET_API_KEYS_FROM_CONFIG=true)
    api_keys: Dict[str, str] = {}
    if args.openai_key:
        api_keys["OPENAI_API_KEY"] = args.openai_key
    if args.anthropic_key:
        api_keys["ANTHROPIC_API_KEY"] = args.anthropic_key
    if args.tavily_key:
        api_keys["TAVILY_API_KEY"] = args.tavily_key
    if api_keys:
        configurable["apiKeys"] = api_keys

    payload: Dict[str, Any] = {
        "input": {"messages": messages},
        "config": {"configurable": configurable},
    }
    return payload


def non_stream_call(base_url: str, graph: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/graphs/{urllib.parse.quote(graph)}/invoke"
    with httpx.Client(timeout=None) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


def stream_call(base_url: str, graph: str, payload: Dict[str, Any], raw: bool = False) -> Optional[Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/api/graphs/{urllib.parse.quote(graph)}/stream"
    final_state: Optional[Dict[str, Any]] = None
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", url, json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    # Non-JSON chunk
                    if raw:
                        print(line)
                    continue
                if raw:
                    print(json.dumps(event))
                    continue

                etype = event.get("event")
                name = event.get("name")
                data = event.get("data") or {}
                if etype == "on_node_start":
                    print(f"▶️  {name}")
                elif etype == "on_node_end":
                    out = data.get("output") or {}
                    if isinstance(out, dict):
                        if out.get("research_brief"):
                            print("🧭 Research Brief:\n", out.get("research_brief"))
                        if out.get("compressed_research"):
                            print("🔎 Research Synthesis:\n", out.get("compressed_research"))
                        if out.get("notes"):
                            for n in out.get("notes") or []:
                                print("📌 Note:", n)
                        if out.get("final_report"):
                            print("📄 Final Report (partial):\n", out.get("final_report"))
                elif etype == "on_chain_end":
                    final_state = data.get("output")
    return final_state


def main() -> None:
    p = argparse.ArgumentParser(description="Call LangGraph server graphs parametrically")
    p.add_argument("--graph", default="Essay Writer", help="Graph name as in langgraph.json")
    p.add_argument("--base-url", default="http://127.0.0.1:2024", help="LangGraph server base URL")
    p.add_argument("--message", help="Human message to send")
    p.add_argument("--input-file", help="Path to a text file as message content")
    p.add_argument("--thread-id", default="cli-session", help="Stable thread/session id")
    p.add_argument("--stream", action="store_true", help="Use streaming events endpoint")
    p.add_argument("--raw-events", action="store_true", help="Print raw JSON events in streaming mode")
    p.add_argument("--output", help="Write final_report to this file")

    # model/search overrides
    p.add_argument("--research-model")
    p.add_argument("--final-model")
    p.add_argument("--summarization-model")
    p.add_argument("--compression-model")
    p.add_argument("--search-api", choices=["tavily", "openai", "anthropic", "none"])

    # agenticity
    p.add_argument("--max-researcher-iterations", type=int)
    p.add_argument("--max-react-tool-calls", type=int)
    p.add_argument("--max-concurrent-research-units", type=int)

    # optional api keys via config
    p.add_argument("--openai-key")
    p.add_argument("--anthropic-key")
    p.add_argument("--tavily-key")

    args = p.parse_args()

    payload = build_payload(args)

    if args.stream:
        final_state = stream_call(args.base_url, args.graph, payload, raw=args.raw_events)
    else:
        final_state = non_stream_call(args.base_url, args.graph, payload)
        print(json.dumps(final_state, indent=2, ensure_ascii=False))

    if not final_state:
        return

    if args.output:
        final_report = (final_state or {}).get("final_report")
        if final_report:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(final_report)
            print(f"Saved final report to {args.output}")


if __name__ == "__main__":
    main()

