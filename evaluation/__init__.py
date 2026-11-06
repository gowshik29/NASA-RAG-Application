# DeepEval evaluation engine for NASA-ORBIT (deepeval >= 4.0)
"""Provides a thin wrapper around the `deepeval` library using a local Ollama model.
Metrics are measured via LLMTestCase as required by deepeval 4.x.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional

from config import settings

# DeepEval 4.x model wrapper
# Optional import of deepeval – will be lazy-loaded when needed
try:
    from deepeval.models import OllamaModel
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        HallucinationMetric,
        SummarizationMetric,
        BiasMetric,
        ToxicityMetric,
    )
except ImportError as deepeval_import_error:  # pragma: no cover
    # Define minimal stubs so that the module imports without deepeval installed.
    class _MissingDeepeval:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "DeepEval is unavailable or incompatible. Install the supported version "
                "with 'python -m pip install -r requirements.txt' or disable evaluation features. "
                f"Import error: {deepeval_import_error}"
            ) from deepeval_import_error
    OllamaModel = _MissingDeepeval
    LLMTestCase = _MissingDeepeval
    FaithfulnessMetric = _MissingDeepeval
    AnswerRelevancyMetric = _MissingDeepeval
    ContextualPrecisionMetric = _MissingDeepeval
    ContextualRecallMetric = _MissingDeepeval
    ContextualRelevancyMetric = _MissingDeepeval
    HallucinationMetric = _MissingDeepeval
    SummarizationMetric = _MissingDeepeval
    BiasMetric = _MissingDeepeval
    ToxicityMetric = _MissingDeepeval

# --------------------------------------------------------------------------
# DeepEval engine exposing a convenient evaluate() method
# --------------------------------------------------------------------------
class DeepEvalEngine:
    """Run a suite of DeepEval metrics on a generated answer.

    Parameters
    ----------
    enabled_metrics: list[str] | None
        Optional whitelist of metric names. If ``None`` all supported metrics
        are executed.
    model_name: str | None
        Ollama model name. Defaults to ``settings.ollama_eval_model``.
    """

    DEFAULT_METRICS = [
        "faithfulness",
        "answer_relevancy",
        "contextual_precision",
        "contextual_recall",
        "contextual_relevancy",
        "hallucination",
        "summarization",
        "bias",
        "toxicity",
    ]

    METRIC_LABELS = {
        "faithfulness":          "Faithfulness",
        "answer_relevancy":      "Answer Relevancy",
        "contextual_precision":  "Contextual Precision",
        "contextual_recall":     "Contextual Recall",
        "contextual_relevancy":  "Contextual Relevancy",
        "hallucination":         "Hallucination",
        "summarization":         "Summarization",
        "bias":                  "Bias",
        "toxicity":              "Toxicity",
    }

    METRIC_DESCRIPTIONS = {
        "faithfulness":         "Measures whether the answer is factually consistent with the retrieved context. High = answer sticks to evidence.",
        "answer_relevancy":     "Measures whether the answer directly addresses the question. High = answer is on-topic.",
        "contextual_precision": "Measures whether the retrieved contexts are relevant and ordered with the most relevant first.",
        "contextual_recall":    "Measures whether all relevant information from the ground truth is covered by the retrieved context.",
        "contextual_relevancy": "Measures how relevant the retrieved context is to the query overall.",
        "hallucination":        "Detects content in the answer not supported by context. Low = fewer hallucinations (inverted scale).",
        "summarization":        "Measures whether the answer is a good summary: covers key facts without adding noise.",
        "bias":                 "Detects political, gender, racial or other biases in the answer. Low = less bias.",
        "toxicity":             "Detects toxic, offensive or harmful language in the answer. Low = safer output.",
    }

    # Thresholds for pass/fail colouring in the UI
    PASS_THRESHOLDS = {
        "faithfulness":          0.7,
        "answer_relevancy":      0.7,
        "contextual_precision":  0.7,
        "contextual_recall":     0.7,
        "contextual_relevancy":  0.7,
        "hallucination":         0.5,   # lower is better
        "summarization":         0.7,
        "bias":                  0.3,   # lower is better
        "toxicity":              0.3,   # lower is better
    }

    # Metrics where a LOWER score is better
    LOWER_IS_BETTER = {"hallucination", "bias", "toxicity"}

    def __init__(
        self,
        enabled_metrics: Optional[List[str]] = None,
        model_name: Optional[str] = None,
    ):
        model = model_name or settings.ollama_eval_model
        self.llm = OllamaModel(model=model, base_url=settings.ollama_base_url)
        self.enabled = enabled_metrics or self.DEFAULT_METRICS

    def _build_metric(self, name: str):
        """Instantiate a deepeval metric by name."""
        common = dict(model=self.llm, include_reason=True, async_mode=False)
        if name == "faithfulness":
            return FaithfulnessMetric(threshold=0.7, **common)
        if name == "answer_relevancy":
            return AnswerRelevancyMetric(threshold=0.7, **common)
        if name == "contextual_precision":
            return ContextualPrecisionMetric(threshold=0.7, **common)
        if name == "contextual_recall":
            return ContextualRecallMetric(threshold=0.7, **common)
        if name == "contextual_relevancy":
            return ContextualRelevancyMetric(threshold=0.7, **common)
        if name == "hallucination":
            return HallucinationMetric(threshold=0.5, **common)
        if name == "summarization":
            return SummarizationMetric(threshold=0.7, **common)
        if name == "bias":
            return BiasMetric(threshold=0.3, **common)
        if name == "toxicity":
            return ToxicityMetric(threshold=0.3, **common)
        return None

    def _run_metric(
        self,
        name: str,
        *,
        answer: str,
        context: List[str],
        question: str,
        expected_output: Optional[str] = None,
    ) -> Dict[str, Any]:
        metric = self._build_metric(name)
        if metric is None:
            return {"score": None, "reason": "Metric not found", "success": None}
        try:
            tc = LLMTestCase(
                input=question,
                actual_output=answer,
                expected_output=expected_output or answer,
                retrieval_context=context if context else ["No context provided."],
                context=context if context else ["No context provided."],
            )
            metric.measure(tc)
            return {
                "score":   float(metric.score) if metric.score is not None else None,
                "reason":  getattr(metric, "reason", None),
                "success": metric.is_successful(),
            }
        except Exception as exc:
            return {"score": None, "reason": str(exc), "success": None}

    def evaluate(
        self,
        *,
        answer: str,
        context: List[str],
        question: str,
        expected_output: Optional[str] = None,
        enabled_metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Execute all (or selected) enabled metrics.

        Returns a dict mapping metric name → {score, reason, success}.
        An 'overall' key holds the mean of all valid scores.
        """
        metrics_to_run = enabled_metrics or self.enabled
        results: Dict[str, Dict[str, Any]] = {}
        for name in metrics_to_run:
            results[name] = self._run_metric(
                name,
                answer=answer,
                context=context,
                question=question,
                expected_output=expected_output,
            )
        valid = [v["score"] for v in results.values() if isinstance(v["score"], float)]
        results["overall"] = {
            "score":   round(sum(valid) / len(valid), 4) if valid else None,
            "reason":  f"Mean of {len(valid)} metric(s).",
            "success": None,
        }
        return results

    @staticmethod
    def markdown_report(scores: Dict[str, Dict[str, Any]]) -> str:
        lines = ["### 📊 DeepEval Evaluation Report", ""]
        for name, data in scores.items():
            label = DeepEvalEngine.METRIC_LABELS.get(name, name.replace("_", " ").title())
            score = data.get("score")
            reason = data.get("reason", "")
            if isinstance(score, float):
                lines.append(f"* **{label}:** {score:.3f} — _{reason}_")
            else:
                lines.append(f"* **{label}:** N/A")
        return "\n".join(lines)


# Backwards-compatible alias used by legacy code
def run_evaluation(answer: str, context: str, question: str) -> Dict[str, Any]:
    """Thin wrapper kept for backwards compatibility with older callers."""
    engine = DeepEvalEngine()
    ctx_list = [context] if isinstance(context, str) else context
    return engine.evaluate(answer=answer, context=ctx_list, question=question)


# Quick sanity check
if __name__ == "__main__":
    q = "What increase in solar-wind speed was reported?"
    ctx = "[Source: example.pdf, p.3] The solar-wind speed rose by 15% during the event."
    ans = "The solar-wind speed rose by 15% during the event."
    engine = DeepEvalEngine(enabled_metrics=["faithfulness", "answer_relevancy"])
    rpt = engine.evaluate(answer=ans, context=[ctx], question=q)
    print(DeepEvalEngine.markdown_report(rpt))
