from exam_parser import markdown_pipeline as pipeline
from exam_parser.fidelity_retry_guard import ensure_condition_fidelity_without_retry
from exam_parser.models import ExtractedTask


class Client:
    provider_name = "Test"

    def extract_markdown(self, *_args, **_kwargs):
        raise AssertionError("fidelity guard must not make a second LLM request")


def test_fidelity_mismatch_falls_back_to_source_without_retry() -> None:
    source = "Найдите объём, если $AB=3$, $AD=3$, $AA_1=4$."
    candidate = "Найдите объём, если $A=3$, $AD=3$, $AA_1=4$."
    task = ExtractedTask(task_num="8", condition=candidate)

    result = ensure_condition_fidelity_without_retry(
        pipeline,
        Client(),
        task,
        source,
    )

    assert result.task_num == "8"
    assert result.condition == pipeline._normalize_condition_artifacts(
        source,
        task_num="8",
    )


def test_fidelity_match_keeps_model_condition_without_retry() -> None:
    source = "Решите уравнение $x^2=4$."
    task = ExtractedTask(task_num="13", condition=source)

    result = ensure_condition_fidelity_without_retry(
        pipeline,
        Client(),
        task,
        source,
    )

    assert result.task_num == "13"
    assert result.condition == source
