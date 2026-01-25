"""
DA-Code Evaluator

Evaluates data science task outputs, returns score (0.0-1.0).

Evaluation Types:
- compare_csv: CSV column/row comparison with tolerance
- compare_text: JSON/dictionary value comparison
- compare_ml: ML metric evaluation (f1/accuracy/etc), normalized to [0,1]
- compare_image: Image + data array + chart attributes evaluation
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Add DA-Code path for metrics import
DACODE_PATH = Path(__file__).parent.parent.parent.parent / "datasets" / "dacode"
if str(DACODE_PATH) not in sys.path:
    sys.path.insert(0, str(DACODE_PATH))


# ============================================================================
# Metric Definitions - Objective descriptions for ReCreate-Agent
# ============================================================================

METRIC_DEFINITIONS = {
    "compare_csv": {
        "name": "CSV Comparison",
        "definition": "Compares predicted CSV against gold standard CSV file(s). "
                     "Score is based on column-wise matching with numerical tolerance (1e-3). "
                     "Options: 'ignore_order' ignores row order, 'condition_cols' specifies columns to compare, "
                     "'score_rule' can be 'all' (exact match) or 'divide' (partial credit).",
    },
    "compare_text": {
        "name": "JSON/Text Comparison",
        "definition": "Compares predicted JSON dictionary against expected values. "
                     "Score based on key-value matching with numerical tolerance (1e-3). "
                     "Keys are case-insensitive. Options: 'score_rule' can be 'all' or 'devide'.",
    },
    "compare_ml": {
        "name": "Machine Learning Metric",
        "definition": "Evaluates ML model predictions using specified metric. "
                     "Raw metric is normalized: score = (raw - lower_bound) / (upper_bound - lower_bound). "
                     "Common metrics: f1, accuracy, silhouette, rmsle, mae.",
    },
    "compare_image": {
        "name": "Visualization Comparison",
        "definition": "Evaluates chart/plot output through multiple checks: "
                     "(1) Image pixel comparison, (2) Data array (result.npy) comparison, "
                     "(3) Chart attributes (title, labels, colors, etc.) from plot.json. "
                     "Score is 1.0 if image matches OR (data matches AND all specified attributes match).",
    },
    "compare_competition_ml": {
        "name": "ML Competition Metric",
        "definition": "Evaluates Kaggle-style ML competition submissions. "
                     "Similar to compare_ml but for regression/ranking tasks. "
                     "Common metrics: rmsle, mae, rmse. Score normalized by bounds.",
    },
    "compare_sqlite": {
        "name": "SQLite Database Comparison",
        "definition": "Compares SQLite database output against gold standard. "
                     "Evaluates table structure, data content, and query results.",
    },
}


@dataclass
class DACodeResult:
    """
    DA-Code evaluation result.
    
    Core: score (0.0-1.0) + metric definition.
    """
    instance_id: str
    score: float = 0.0
    
    # Task metadata
    task_type: str = ""             # statistical analysis, machine learning, etc.
    result_type: str = ""           # csv, text, binary classification, bar, etc.
    hardness: str = ""              # Easy, Medium, Hard
    
    # Evaluation method
    metric_func: str = ""           # compare_csv, compare_text, compare_ml, compare_image
    metric_definition: str = ""     # What the metric measures
    
    # ML-specific (when metric_func == "compare_ml")
    ml_metric: str = ""             # f1, accuracy, silhouette, etc.
    ml_raw_score: float = 0.0       # Raw metric value before normalization
    ml_upper_bound: float = 0.0
    ml_lower_bound: float = 0.0
    
    # Image-specific check results (when metric_func == "compare_image")
    image_checks: dict = field(default_factory=dict)
    
    # File information
    output_files: list[str] = field(default_factory=list)
    expected_files: list[str] = field(default_factory=list)
    
    # Evaluation options used
    eval_options: dict = field(default_factory=dict)
    
    # Raw result from metrics function
    raw_result: dict = field(default_factory=dict)
    
    # Error (if evaluation failed)
    error: str = ""


class DACodeEvaluator:
    """DA-Code evaluator using native metrics functions."""
    
    def __init__(
        self,
        gold_dir: str | Path = "datasets/dacode/da_code/gold/gold",
        eval_config_dir: str | Path = "datasets/dacode/da_code/configs/eval",
    ):
        self.gold_dir = Path(gold_dir)
        self.eval_config_dir = Path(eval_config_dir)
        self._metrics = None
    
    @property
    def metrics(self):
        if self._metrics is None:
            from da_agent.evaluators import metrics
            self._metrics = metrics
        return self._metrics
    
    def load_eval_config(self, instance_id: str) -> dict | None:
        """Load evaluation config from jsonl files."""
        for eval_file in self.eval_config_dir.glob("eval_*.jsonl"):
            for line in eval_file.read_text().strip().split("\n"):
                if line.strip():
                    config = json.loads(line)
                    if config.get("id") == instance_id:
                        return config
        return None
    
    def evaluate_in_container(
        self,
        container_id: str,
        instance: dict,
        output_dir: Path,
        timeout: int = 60,
    ) -> DACodeResult:
        """Evaluate task output in container."""
        instance_id = instance.get("id", instance.get("instance_id", "unknown"))
        result = DACodeResult(instance_id=instance_id)
        
        try:
            # Load eval config
            eval_config = self.load_eval_config(instance_id)
            if not eval_config:
                result.error = f"Evaluation config not found for {instance_id}"
                return result
            
            config = eval_config.get("config", {})
            result.task_type = config.get("task", "")
            result.result_type = config.get("type", "")
            result.hardness = config.get("hardness", "")
            
            # Copy output from container
            self._copy_output_from_container(container_id, output_dir)
            
            # Get evaluation method
            func_names = eval_config.get("func", ["compare_csv"])
            if isinstance(func_names, str):
                func_names = [func_names]
            result.metric_func = func_names[0]
            
            # Set metric definition
            metric_info = METRIC_DEFINITIONS.get(result.metric_func, {})
            result.metric_definition = metric_info.get("definition", "")
            
            # Get evaluation params
            expected = eval_config.get("result", [{}])
            if isinstance(expected, dict):
                expected = [expected]
            expected = expected[0] if expected else {}
            
            options = eval_config.get("options", [{}])
            if isinstance(options, dict):
                options = [options]
            options = options[0] if options else {}
            result.eval_options = options
            
            gold_dir = self.gold_dir / instance_id
            
            # Run evaluation
            self._run_evaluation(result, output_dir, gold_dir, expected, options, config)
            
        except Exception as e:
            result.error = f"Evaluation error: {type(e).__name__}: {e}"
        
        return result
    
    def _copy_output_from_container(self, container_id: str, output_dir: Path):
        """
        Copy output from container.
        
        Always try to copy files from container to ensure we have the latest output.
        Volume mount should sync automatically, but docker cp is a safety fallback.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Always try docker cp to ensure we have the latest files
        # This is important because volume mount sync might have timing issues
        try:
            subprocess.run(
                ["docker", "cp", f"{container_id}:/workspace/.", str(output_dir)],
                check=True, capture_output=True, timeout=30,
            )
        except subprocess.CalledProcessError as e:
            # Container may have been removed or no files to copy - this is OK
            # if volume mount already synced the files
            pass
        except subprocess.TimeoutExpired:
            # Timeout is OK - volume mount should have synced files
            pass
        except Exception:
            # Any other error is also OK - volume mount fallback
            pass
    
    def _run_evaluation(
        self,
        result: DACodeResult,
        output_dir: Path,
        gold_dir: Path,
        expected: dict,
        options: dict,
        config: dict,
    ):
        # Get file lists
        result.output_files = [f.name for f in output_dir.glob("*") if f.is_file()]
        
        files = expected.get("file", [])
        if isinstance(files, str):
            files = [files]
        result.expected_files = files
        
        # Route to specific evaluator
        if result.metric_func == "compare_csv":
            self._eval_csv(result, output_dir, gold_dir, expected, options)
        elif result.metric_func == "compare_text":
            self._eval_text(result, output_dir, gold_dir, expected, options)
        elif result.metric_func == "compare_ml":
            self._eval_ml(result, output_dir, gold_dir, expected, options, config)
        elif result.metric_func == "compare_competition_ml":
            self._eval_competition_ml(result, output_dir, gold_dir, expected, options, config)
        elif result.metric_func == "compare_image":
            self._eval_image(result, output_dir, gold_dir, expected, options)
        elif result.metric_func == "compare_sqlite":
            self._eval_sqlite(result, output_dir, gold_dir, expected, options)
        else:
            result.error = f"Unsupported evaluation function: {result.metric_func}"
    
    def _eval_csv(self, result: DACodeResult, output_dir: Path, gold_dir: Path,
                  expected: dict, options: dict):
        files = expected.get("file", ["result.csv"])
        if isinstance(files, str):
            files = [files]
        
        output_file = output_dir / files[0]
        gold_files = [gold_dir / f for f in files]
        
        if not output_file.exists():
            result.score = 0.0
            result.error = f"Output file not found: {files[0]}"
            return
        
        try:
            score = self.metrics.compare_csv(
                str(output_file),
                [str(g) for g in gold_files] if len(gold_files) > 1 else str(gold_files[0]),
                **options,
            )
            result.score = float(score) if isinstance(score, (int, float)) else 0.0
            result.raw_result = {"score": result.score}
        except Exception as e:
            result.score = 0.0
            result.error = f"CSV comparison failed: {e}"
    
    def _eval_text(self, result: DACodeResult, output_dir: Path, gold_dir: Path,
                   expected: dict, options: dict):
        expected_value = expected.get("number", [])
        output_file = expected.get("file", ["result.json"])
        if isinstance(output_file, list):
            output_file = output_file[0] if output_file else "result.json"
        
        output_path = output_dir / output_file
        
        try:
            raw_result = self.metrics.compare_text(str(output_path), expected_value, **options)
            if isinstance(raw_result, dict):
                result.score = raw_result.get("score", 0.0)
                result.raw_result = raw_result
            else:
                result.score = float(raw_result) if raw_result else 0.0
        except Exception as e:
            result.score = 0.0
            result.error = f"Text comparison failed: {e}"
    
    def _eval_ml(self, result: DACodeResult, output_dir: Path, gold_dir: Path,
                 expected: dict, options: dict, config: dict):
        files = expected.get("file", "result.csv")
        if isinstance(files, list):
            files = files[0]
        
        output_file = output_dir / files
        gold_file = gold_dir / files
        
        # ML config
        result.ml_metric = config.get("metric", "f1")
        result.ml_upper_bound = config.get("upper_bound", 1.0)
        result.ml_lower_bound = config.get("lower_bound", 0.0)
        
        if not output_file.exists():
            result.score = 0.0
            result.error = f"Output file not found: {files}"
            return
        
        try:
            raw_result = self.metrics.compare_ml(
                str(output_file), str(gold_file), config=config, **options
            )
            result.raw_result = raw_result
            result.score = raw_result.get("score", 0.0)
            
            # Calculate raw score before normalization
            if result.ml_upper_bound != result.ml_lower_bound:
                result.ml_raw_score = (
                    result.score * (result.ml_upper_bound - result.ml_lower_bound)
                    + result.ml_lower_bound
                )
            
            if raw_result.get("errors"):
                result.error = "; ".join(raw_result["errors"])
                
        except Exception as e:
            result.score = 0.0
            result.error = f"ML evaluation failed: {e}"
    
    def _eval_image(self, result: DACodeResult, output_dir: Path, gold_dir: Path,
                    expected: dict, options: dict):
        files = expected.get("file", [])
        if isinstance(files, str):
            files = [files]
        
        # Collect output files
        output_images = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
        output_json = list(output_dir.glob("dabench/plot.json")) + list(output_dir.glob("plot.json"))
        output_npy = list(output_dir.glob("dabench/result.npy")) + list(output_dir.glob("result.npy"))
        
        result_files = (
            [str(f) for f in output_images] +
            [str(f) for f in output_json] +
            [str(f) for f in output_npy]
        )
        
        gold_files = [gold_dir / f for f in files]
        gold_files_str = [str(f) for f in gold_files if f.exists()]
        
        if not result_files:
            result.score = 0.0
            result.error = "No output files found (expected: image, result.npy, plot.json)"
            return
        
        try:
            raw_result = self.metrics.compare_image(result_files, gold_files_str, **options)
            result.raw_result = raw_result
            result.score = raw_result.get("score", 0.0)
            result.image_checks = {k: v for k, v in raw_result.items() if k != "score"}
        except Exception as e:
            result.score = 0.0
            result.error = f"Image evaluation failed: {e}"
    
    def _eval_competition_ml(self, result: DACodeResult, output_dir: Path, gold_dir: Path,
                             expected: dict, options: dict, config: dict):
        """Evaluate ML competition tasks (e.g., Kaggle-style submissions)."""
        files = expected.get("file", "result.csv")
        if isinstance(files, list):
            files = files[0]
        
        output_file = output_dir / files
        gold_file = gold_dir / files
        
        # ML config
        result.ml_metric = config.get("metric", "rmsle")
        result.ml_upper_bound = config.get("upper_bound", 1.0)
        result.ml_lower_bound = config.get("lower_bound", 0.0)
        
        if not output_file.exists():
            result.score = 0.0
            result.error = f"Output file not found: {files}"
            return
        
        try:
            raw_result = self.metrics.compare_competition_ml(
                str(output_file), str(gold_file), config=config, **options
            )
            result.raw_result = raw_result
            result.score = raw_result.get("score", 0.0)
            
            # Store raw score info
            if "raw_score" in raw_result:
                result.ml_raw_score = raw_result["raw_score"]
            
            # Capture any errors from evaluation
            if "errors" in raw_result and raw_result["errors"]:
                result.error = "; ".join(raw_result["errors"][:3])  # First 3 errors
        except Exception as e:
            result.score = 0.0
            result.error = f"Competition ML evaluation failed: {e}"
    
    def _eval_sqlite(self, result: DACodeResult, output_dir: Path, gold_dir: Path,
                     expected: dict, options: dict):
        """Evaluate SQLite database outputs."""
        files = expected.get("file", "result.db")
        if isinstance(files, list):
            files = files[0]
        
        output_file = output_dir / files
        gold_file = gold_dir / files
        
        if not output_file.exists():
            result.score = 0.0
            result.error = f"Output file not found: {files}"
            return
        
        try:
            score = self.metrics.compare_sqlite(str(output_file), str(gold_file), **options)
            result.score = float(score) if isinstance(score, (int, float)) else 0.0
        except Exception as e:
            result.score = 0.0
            result.error = f"SQLite comparison failed: {e}"
    
    def evaluate_local(self, instance_id: str, output_dir: Path) -> DACodeResult:
        """Local evaluation (no container)."""
        result = DACodeResult(instance_id=instance_id)
        
        eval_config = self.load_eval_config(instance_id)
        if not eval_config:
            result.error = f"Evaluation config not found for {instance_id}"
            return result
        
        config = eval_config.get("config", {})
        result.task_type = config.get("task", "")
        result.result_type = config.get("type", "")
        result.hardness = config.get("hardness", "")
        
        func_names = eval_config.get("func", ["compare_csv"])
        result.metric_func = func_names[0] if isinstance(func_names, list) else func_names
        
        metric_info = METRIC_DEFINITIONS.get(result.metric_func, {})
        result.metric_definition = metric_info.get("definition", "")
        
        expected = eval_config.get("result", [{}])
        if isinstance(expected, list):
            expected = expected[0]
        
        options = eval_config.get("options", [{}])
        if isinstance(options, list):
            options = options[0]
        result.eval_options = options
        
        gold_dir = self.gold_dir / instance_id
        
        try:
            self._run_evaluation(result, output_dir, gold_dir, expected, options, config)
        except Exception as e:
            result.error = str(e)
        
        return result


def format_dacode_result_for_recreate_agent(result: DACodeResult) -> str:
    """
    Format DA-Code result for ReCreate-Agent.
    
    Provides: score + metric definition. Analysis is left to ReCreate-Agent.
    """
    lines = [
        "=" * 60,
        "DA-CODE EVALUATION RESULT",
        "=" * 60,
        "",
        "## Task",
        f"- Instance: {result.instance_id}",
        f"- Type: {result.task_type}",
        f"- Format: {result.result_type}",
        f"- Difficulty: {result.hardness}",
        "",
        "## Score",
        f"**{result.score:.4f}** (range: 0.0 - 1.0)",
        "",
    ]
    
    if result.error:
        lines.extend([
            "## Error",
            result.error,
            "",
        ])
    
    # Metric definition
    lines.extend([
        "## Evaluation Method",
        f"Function: `{result.metric_func}`",
        "",
        f"{result.metric_definition}",
        "",
    ])
    
    # ML-specific info
    if result.metric_func == "compare_ml" and result.ml_metric:
        lines.extend([
            "## ML Metric Details",
            f"- Metric: {result.ml_metric}",
            f"- Raw value: {result.ml_raw_score:.4f}",
            f"- Normalization range: [{result.ml_lower_bound}, {result.ml_upper_bound}]",
            f"- Formula: score = (raw - {result.ml_lower_bound}) / ({result.ml_upper_bound} - {result.ml_lower_bound})",
            "",
        ])
    
    # Image check results
    if result.metric_func == "compare_image" and result.image_checks:
        lines.extend([
            "## Image Check Results",
        ])
        for key, passed in result.image_checks.items():
            status = "PASS" if passed else "FAIL"
            lines.append(f"- {key}: {status}")
        lines.append("")
    
    # File info
    lines.extend([
        "## Files",
        f"- Output: {result.output_files}",
        f"- Expected: {result.expected_files}",
    ])
    
    if result.eval_options:
        lines.extend([
            "",
            "## Evaluation Options",
            f"{json.dumps(result.eval_options, indent=2)}",
        ])
    
    # Raw evaluation result (for debugging/understanding)
    if result.raw_result and result.score < 1.0:
        lines.extend([
            "",
            "## Raw Evaluation Details",
        ])
        # Format raw_result, filter out overly long values
        for key, value in result.raw_result.items():
            if key == "score":
                continue  # Already displayed
            str_value = str(value)
            if len(str_value) > 200:
                str_value = str_value[:200] + "..."
            lines.append(f"- {key}: {str_value}")
    
    return "\n".join(lines)


# Convenience function
def run_dacode_evaluation(
    container_id: str,
    instance: dict,
    output_dir: Path,
    gold_dir: str | Path = "datasets/dacode/da_code/gold/gold",
) -> DACodeResult:
    """Run DA-Code evaluation."""
    evaluator = DACodeEvaluator(gold_dir=gold_dir)
    return evaluator.evaluate_in_container(container_id, instance, output_dir)
