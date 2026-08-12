import json
from types import SimpleNamespace

from tests import extract_langsmith_data


class FakeClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def read_project(self, project_name):
        return SimpleNamespace(reference_dataset_id="dataset-id")

    def list_examples(self, dataset_id):
        return [
            SimpleNamespace(
                id="known-example",
                metadata={"id": "benchmark-example-1"},
            )
        ]

    def list_runs(self, **kwargs):
        return [
            SimpleNamespace(
                reference_example_id="known-example",
                inputs={
                    "inputs": {
                        "messages": [{"content": "matched prompt"}],
                    }
                },
                outputs={"final_report": "matched report"},
            ),
            SimpleNamespace(
                reference_example_id="unmatched-example",
                inputs={
                    "inputs": {
                        "messages": [{"content": "unmatched prompt"}],
                    }
                },
                outputs={"final_report": "unmatched report"},
            ),
        ]


def test_export_skips_runs_without_reference_examples(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setattr(extract_langsmith_data, "Client", FakeClient)
    monkeypatch.chdir(tmp_path)

    output_path = extract_langsmith_data.extract_langsmith_data(
        project_name="experiment",
        model_name="model",
        dataset_name="dataset",
        api_key="test-key",
    )

    records = [
        json.loads(line)
        for line in (tmp_path / output_path).read_text().splitlines()
    ]
    assert records == [
        {
            "id": "benchmark-example-1",
            "prompt": "matched prompt",
            "article": "matched report",
        }
    ]
    assert "Skipped 1 run(s) without a matching reference example" in capsys.readouterr().out
