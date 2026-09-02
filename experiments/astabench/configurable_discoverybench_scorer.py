"""Rescore DiscoveryBench logs with an explicit, auditable judge model.

The AstaBench 0.5.3 scorer hard-codes ``gpt-4o-2024-08-06``. This wrapper keeps
the official HMS implementation unchanged while making the judge model an
explicit scorer argument. Scores from any other model must be labelled as
compatible HMS scores rather than official AstaBench scores.
"""

from __future__ import annotations

from pydantic import ValidationError

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from astabench.evals.discoverybench.eval_utils import (
    run_eval_gold_vs_gen_NL_hypo_workflow,
)
from astabench.evals.discoverybench.task import DiscoveryBenchOutput, extract_output


OFFICIAL_JUDGE_MODEL = "gpt-4o-2024-08-06"


@scorer(metrics=[mean(), stderr()])
def score_discoverybench_configurable(
    judge_model: str = OFFICIAL_JUDGE_MODEL,
) -> Scorer:
    """Return the official HMS scorer logic with an explicit judge model."""

    async def score(state: TaskState, target: Target) -> Score:
        try:
            output = extract_output(state)
        except Exception as exc:
            if isinstance(exc, ValidationError):
                message = "Invalid output format"
            elif isinstance(exc, ValueError):
                message = "Failed to extract output"
            else:
                message = "Error parsing agent output"
            return Score(
                value=0.0,
                answer=state.output.completion,
                explanation=f"{message}: {exc}",
                metadata={
                    "judge_model": judge_model,
                    "official_judge": judge_model == OFFICIAL_JUDGE_MODEL,
                },
            )

        if isinstance(output, DiscoveryBenchOutput):
            generated_hypothesis = output.hypothesis
            generated_workflow = output.workflow
        else:
            generated_hypothesis = output
            generated_workflow = ""

        result = await run_eval_gold_vs_gen_NL_hypo_workflow(
            query=state.metadata["query"],
            gold_hypo=target[0],
            gold_workflow=target[1],
            gen_hypo=generated_hypothesis,
            gen_workflow=generated_workflow,
            dataset_meta=state.metadata["metadata"],
            llm_used=judge_model,
            use_column_metadata=True,
        )
        result["judge_model"] = judge_model
        result["official_judge"] = judge_model == OFFICIAL_JUDGE_MODEL

        return Score(
            value=result["HMS"],
            answer=state.output.completion,
            metadata=result,
        )

    return score


__all__ = [
    "OFFICIAL_JUDGE_MODEL",
    "score_discoverybench_configurable",
]
