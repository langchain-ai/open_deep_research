SERVER ?= http://127.0.0.1:2024
GRAPH ?= Essay Writer
MSG ?= Hello world
THREAD ?= $(shell date +%s)
OUT ?=
RESEARCH_MODEL ?=
FINAL_MODEL ?=
SEARCH_API ?=

.PHONY: dev invoke stream essay essay-stream deep deep-stream compare-models clean

dev:
	uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking

invoke:
	uv run python scripts/call_api.py --base-url "$(SERVER)" --graph "$(GRAPH)" --message "$(MSG)" --thread-id "$(THREAD)" \
	$(if $(RESEARCH_MODEL),--research-model "$(RESEARCH_MODEL)") \
	$(if $(FINAL_MODEL),--final-model "$(FINAL_MODEL)") \
	$(if $(SEARCH_API),--search-api "$(SEARCH_API)") \
	$(if $(OUT),--output "$(OUT)")

stream:
	uv run python scripts/call_api.py --base-url "$(SERVER)" --graph "$(GRAPH)" --message "$(MSG)" --thread-id "$(THREAD)" --stream \
	$(if $(RESEARCH_MODEL),--research-model "$(RESEARCH_MODEL)") \
	$(if $(FINAL_MODEL),--final-model "$(FINAL_MODEL)") \
	$(if $(SEARCH_API),--search-api "$(SEARCH_API)")

essay: GRAPH=Essay Writer
essay: invoke

essay-stream: GRAPH=Essay Writer
essay-stream: stream

deep: GRAPH=Deep Researcher
deep: invoke

deep-stream: GRAPH=Deep Researcher
deep-stream: stream

compare-models:
	mkdir -p out
	$(MAKE) invoke GRAPH="$(GRAPH)" MSG="$(MSG)" THREAD="$(THREAD)-A" RESEARCH_MODEL="$(RESEARCH_MODEL)" FINAL_MODEL="$(FINAL_MODEL)" OUT="out/run_A.md"
	$(MAKE) invoke GRAPH="$(GRAPH)" MSG="$(MSG)" THREAD="$(THREAD)-B" RESEARCH_MODEL="$(RESEARCH_MODEL_B)" FINAL_MODEL="$(FINAL_MODEL_B)" OUT="out/run_B.md"
	@echo "Saved out/run_A.md and out/run_B.md; run: diff -u out/run_A.md out/run_B.md || true"

clean:
	rm -rf out

