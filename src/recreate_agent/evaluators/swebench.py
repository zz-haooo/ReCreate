"""
SWE-bench Official Evaluator Wrapper

Uses SWE-bench harness for standard evaluation.

Two evaluation modes:
1. run_swebench_evaluation(): Launch new container (slow, better isolation)
2. run_swebench_in_container(): Evaluate in existing container (fast, reuses container)
"""

import json
import subprocess
import tempfile
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.grading import get_eval_tests_report
    SWEBENCH_AVAILABLE = True
except ImportError:
    SWEBENCH_AVAILABLE = False


@dataclass
class SWEBenchResult:
    """SWE-bench evaluation result."""
    instance_id: str
    resolved: bool = False
    
    # Detailed test results
    fail_to_pass_success: list[str] = field(default_factory=list)  # Previously failing, now passing
    fail_to_pass_failure: list[str] = field(default_factory=list)  # Previously failing, still failing
    pass_to_pass_success: list[str] = field(default_factory=list)  # Not broken
    pass_to_pass_failure: list[str] = field(default_factory=list)  # Broken (regression)
    
    # Raw data
    patch: str = ""
    report_json: dict = field(default_factory=dict)
    test_output: str = ""
    error: str = ""
    
    @property
    def tests_passed(self) -> int:
        return len(self.fail_to_pass_success) + len(self.pass_to_pass_success)
    
    @property
    def tests_failed(self) -> int:
        return len(self.fail_to_pass_failure) + len(self.pass_to_pass_failure)
    
    @property
    def has_regression(self) -> bool:
        """Whether regression bugs were introduced."""
        return len(self.pass_to_pass_failure) > 0
    
    @property
    def fix_incomplete(self) -> bool:
        """Whether the fix is incomplete."""
        return len(self.fail_to_pass_failure) > 0


def run_swebench_in_container(
    container_id: str,
    instance: dict,
    patch: str,
    output_dir: Path,
    timeout: int = 300,
) -> SWEBenchResult:
    """
    Run SWE-bench official evaluation in existing container (fast mode).
    
    Reuses Agent's container, directly runs official eval.sh script.
    Much faster than run_swebench_evaluation() as no new container is needed.
    
    Args:
        container_id: Agent's container ID
        instance: SWE-bench instance data (with FAIL_TO_PASS, PASS_TO_PASS, etc.)
        patch: Agent-generated patch
        output_dir: Output directory
        timeout: Timeout in seconds
    
    Returns:
        SWEBenchResult: Evaluation result
    """
    instance_id = instance.get("instance_id", "unknown")
    result = SWEBenchResult(instance_id=instance_id, patch=patch)
    
    if not SWEBENCH_AVAILABLE:
        result.error = "SWE-bench not installed"
        return result
    
    if not patch or not patch.strip():
        result.error = "Empty patch"
        return result
    
    try:
        # 1. Generate official eval.sh script
        test_spec = make_test_spec(instance)
        eval_script = test_spec.eval_script
        
        # Save eval.sh for debugging
        eval_file = output_dir / "eval.sh"
        eval_file.write_text(eval_script)
        
        # 2. Prepare environment in container
        patch_file = output_dir / "agent.patch"
        patch_file.write_text(patch)
        
        # Copy files to container
        subprocess.run(
            ["docker", "cp", str(patch_file), f"{container_id}:/tmp/agent.patch"],
            check=True, capture_output=True
        )
        subprocess.run(
            ["docker", "cp", str(eval_file), f"{container_id}:/eval.sh"],
            check=True, capture_output=True
        )
        
        # 3. Run evaluation in container
        eval_cmd = """
cd /testbed
git reset --hard HEAD
git clean -fd
git apply /tmp/agent.patch || echo "PATCH_APPLY_FAILED"
chmod +x /eval.sh
/eval.sh 2>&1
"""
        proc = subprocess.run(
            ["docker", "exec", container_id, "bash", "-c", eval_cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        test_output = proc.stdout + proc.stderr
        result.test_output = test_output[-50000:]  # Keep last 50K
        
        # 4. Parse test results
        _parse_test_output(result, test_output, test_spec, output_dir)
        
    except subprocess.TimeoutExpired:
        result.error = f"Evaluation timed out after {timeout}s"
    except subprocess.CalledProcessError as e:
        result.error = f"Docker command failed: {e.stderr[:500] if e.stderr else str(e)}"
    except Exception as e:
        result.error = f"Evaluation error: {type(e).__name__}: {e}"
    
    return result


def _parse_test_output(result: SWEBenchResult, test_output: str, test_spec, output_dir: Path) -> None:
    """Parse test output and extract passed/failed tests."""
    try:
        # Save test output to file (get_logs_eval needs file path)
        log_file = output_dir / "test_output.txt"
        log_file.write_text(test_output)
        
        # Save expected tests info for ReCreate-Agent
        expected_tests_file = output_dir / "expected_tests.txt"
        expected_tests_content = f"""# SWE-bench Expected Tests
# These tests define what "success" means for this issue

## FAIL_TO_PASS (must pass after fix)
{chr(10).join(test_spec.FAIL_TO_PASS) if hasattr(test_spec, 'FAIL_TO_PASS') and test_spec.FAIL_TO_PASS else 'None'}

## PASS_TO_PASS (must not regress)
{chr(10).join(test_spec.PASS_TO_PASS) if hasattr(test_spec, 'PASS_TO_PASS') and test_spec.PASS_TO_PASS else 'None'}
"""
        expected_tests_file.write_text(expected_tests_content)
        
        # Save test patch for ReCreate-Agent (if exists)
        if hasattr(test_spec, 'test_patch') and test_spec.test_patch:
            test_patch_file = output_dir / "test_patch.txt"
            test_patch_file.write_text(f"""# SWE-bench Test Patch
# These tests were added by the benchmark (may not exist in original codebase)

{test_spec.test_patch}
""")
        
        # Use SWE-bench official grading function
        from swebench.harness.grading import get_logs_eval, get_eval_tests_report, EvalType
        from swebench.harness.constants import FAIL_TO_PASS, PASS_TO_PASS, FAIL_ONLY_REPOS
        
        # Parse test log
        eval_status_map, found = get_logs_eval(test_spec, str(log_file))
        
        if not found:
            result.error = "Failed to parse test log (patch may not have applied)"
            return
        
        # Build gold_results
        gold_results = {
            FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
            PASS_TO_PASS: test_spec.PASS_TO_PASS,
        }
        
        # Get evaluation report
        eval_type = EvalType.FAIL_ONLY if test_spec.repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL
        report = get_eval_tests_report(eval_status_map, gold_results, eval_type=eval_type)
        
        result.fail_to_pass_success = report.get("FAIL_TO_PASS", {}).get("success", [])
        result.fail_to_pass_failure = report.get("FAIL_TO_PASS", {}).get("failure", [])
        result.pass_to_pass_success = report.get("PASS_TO_PASS", {}).get("success", [])
        result.pass_to_pass_failure = report.get("PASS_TO_PASS", {}).get("failure", [])
        
        # Resolved: all FAIL_TO_PASS pass, no PASS_TO_PASS failures
        result.resolved = (
            len(result.fail_to_pass_failure) == 0 and
            len(result.pass_to_pass_failure) == 0 and
            len(result.fail_to_pass_success) > 0
        )
        
        result.report_json = report
        
    except Exception as e:
        result.error = f"Failed to parse test output: {e}"
        # Try simple pass/fail detection
        if "PASSED" in test_output or "OK" in test_output:
            result.resolved = "FAILED" not in test_output and "ERROR" not in test_output


def run_swebench_evaluation(
    instance_id: str,
    patch: str,
    output_dir: Path,
    model_name: str = "meta-agent",
    max_workers: int = 1,
    timeout: int = 600,
) -> SWEBenchResult:
    """
    Run SWE-bench official evaluation.
    
    Args:
        instance_id: Instance ID
        patch: Agent-generated patch (git diff format)
        output_dir: Output directory
        model_name: Model name (for evaluation report)
        max_workers: Parallelism
        timeout: Timeout in seconds
    
    Returns:
        SWEBenchResult: Detailed evaluation result
    """
    result = SWEBenchResult(instance_id=instance_id, patch=patch)
    
    if not patch or not patch.strip():
        result.error = "Empty patch"
        return result
    
    # Create temporary prediction file
    predictions = {
        instance_id: {
            "model_name_or_path": model_name,
            "model_patch": patch,
            "instance_id": instance_id,
        }
    }
    
    preds_file = output_dir / f"preds_{instance_id}.json"
    preds_file.write_text(json.dumps(predictions, indent=2))
    
    run_id = f"eval_{instance_id}"
    
    try:
        # Run SWE-bench harness
        cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", "princeton-nlp/SWE-bench_Verified",
            "--predictions_path", str(preds_file),
            "--max_workers", str(max_workers),
            "--run_id", run_id,
        ]
        
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(output_dir),
        )
        
        if proc.returncode != 0:
            result.error = f"Harness failed: {proc.stderr[:500]}"
            return result
        
        # Find and parse report.json
        report_pattern = f"*.{run_id}.json"
        report_files = list(output_dir.glob(report_pattern)) + list(Path(".").glob(report_pattern))
        
        if not report_files:
            # Try logs directory
            log_dir = Path(f"logs/run_evaluation/{run_id}/{model_name}")
            if log_dir.exists():
                instance_dir = log_dir / instance_id
                if instance_dir.exists():
                    report_file = instance_dir / "report.json"
                    if report_file.exists():
                        report_files = [report_file]
        
        if report_files:
            report_file = report_files[0]
            result.report_json = json.loads(report_file.read_text())
            
            # Parse results
            _parse_report(result)
            
            # Try to read test output
            test_output_file = report_file.parent / "test_output.txt"
            if test_output_file.exists():
                result.test_output = test_output_file.read_text()[-10000:]  # Keep last 10K
        else:
            result.error = "Report file not found"
    
    except subprocess.TimeoutExpired:
        result.error = f"Evaluation timed out after {timeout}s"
    except Exception as e:
        result.error = f"Evaluation error: {type(e).__name__}: {e}"
    
    return result


def _parse_report(result: SWEBenchResult):
    """Parse report.json and fill result."""
    report = result.report_json
    
    # Check if resolved
    resolved_dict = report.get("resolved", {})
    if isinstance(resolved_dict, dict):
        result.resolved = resolved_dict.get(result.instance_id, False)
    
    # Parse detailed test results
    if result.instance_id in report:
        instance_report = report[result.instance_id]
    else:
        instance_report = report
    
    # FAIL_TO_PASS
    f2p = instance_report.get("FAIL_TO_PASS", {})
    if isinstance(f2p, dict):
        result.fail_to_pass_success = f2p.get("success", [])
        result.fail_to_pass_failure = f2p.get("failure", [])
    
    # PASS_TO_PASS
    p2p = instance_report.get("PASS_TO_PASS", {})
    if isinstance(p2p, dict):
        result.pass_to_pass_success = p2p.get("success", [])
        result.pass_to_pass_failure = p2p.get("failure", [])


def _extract_error_snippets(lines: list[str], max_lines: int = 200) -> list[str]:
    """
    Generic error extraction based on keyword matching.
    
    Supports: unittest, pytest, doctest, nose, and any framework outputting Exception/Error.
    """
    error_keywords = [
        'FAIL:', 'ERROR:', 'FAILED',  # unittest/Django
        'FAIL', 'ERROR', 'PASSED',    # pytest
        'AssertionError', 'Exception', 'Error:',  # Generic exceptions
        'Traceback', 'raise ', 'assert ',  # Python stack
        'Expected:', 'Got:', 'Failed example:',  # doctest
        'E       ',  # pytest indented error line
        '>>>',  # doctest
        'KeyError', 'ValueError', 'TypeError', 'AttributeError',  # Common exceptions
    ]
    
    result = []
    context_before = 2
    context_after = 5
    
    # Find all lines containing keywords
    key_indices = set()
    for i, line in enumerate(lines):
        if any(kw in line for kw in error_keywords):
            for j in range(max(0, i - context_before), min(len(lines), i + context_after + 1)):
                key_indices.add(j)
    
    # Collect lines in order, deduplicated
    sorted_indices = sorted(key_indices)
    
    # Merge consecutive lines, separate non-consecutive with ...
    prev_i = -2
    for i in sorted_indices:
        if i > prev_i + 1 and result:
            result.append("...")
        result.append(lines[i])
        prev_i = i
        
        if len(result) >= max_lines:
            result.append("... (truncated)")
            break
    
    return result


def _extract_test_error_details(test_output: str, failing_tests: list[str]) -> dict[str, str]:
    """Extract specific error info from test output.
    
    Supports multiple test output formats:
    - Django/unittest: FAIL: test_name (module.class)
    - pytest: test_name ... FAIL
    """
    error_details = {}
    lines = test_output.split('\n')

    for test_name in failing_tests:
        # Extract test method name (without module path)
        short_name = test_name.split()[0] if ' ' in test_name else test_name
        
        test_found = False
        error_lines = []

        for i, line in enumerate(lines):
            # Django/unittest format
            if (line.startswith('FAIL:') or line.startswith('ERROR:')) and short_name in line:
                test_found = True
                start_j = i + 1
                if start_j < len(lines) and lines[start_j].startswith('---'):
                    start_j += 1
                
                for j in range(start_j, min(i + 30, len(lines))):
                    curr_line = lines[j]
                    if (curr_line.startswith('FAIL:') or 
                        curr_line.startswith('ERROR:') or 
                        curr_line.startswith('======') or
                        curr_line.startswith('Ran ') or
                        curr_line.startswith('OK') or
                        curr_line.startswith('FAILED')):
                        break
                    error_lines.append(curr_line)
                break
            
            # pytest format
            elif short_name in line and ('... FAIL' in line or '... ERROR' in line):
                test_found = True
                for j in range(i + 1, min(i + 30, len(lines))):
                    error_lines.append(lines[j])
                    if any(kw in lines[j] for kw in ['... FAIL', '... ERROR', '... ok', 'Ran ']):
                        break
                break

        if test_found and error_lines:
            filtered = [l for l in error_lines if l.strip() and len(l) < 200]
            error_text = '\n'.join(filtered[:15])
            error_details[test_name] = error_text[:1500] if error_text else "Test failed (no details captured)"
        else:
            error_details[test_name] = "No detailed error information found in test output"

    return error_details


def format_swebench_result_for_recreate_agent(result: SWEBenchResult) -> str:
    """Format SWE-bench result for ReCreate-Agent readable text."""
    lines = [
        "=" * 60,
        "SWE-BENCH EVALUATION RESULT",
        "=" * 60,
        "",
        f"**Instance**: {result.instance_id}",
        f"**Resolved**: {'✓ YES' if result.resolved else '✗ NO'}",
        "",
    ]

    if result.error:
        lines.extend([
            "## Evaluation Error",
            result.error,
            "",
        ])
        return "\n".join(lines)
    
    # FAIL_TO_PASS analysis
    lines.extend([
        "## FAIL_TO_PASS Tests (these should now pass after the fix)",
        f"- Now passing: {len(result.fail_to_pass_success)}",
        f"- Still failing: {len(result.fail_to_pass_failure)}",
    ])
    
    if result.fail_to_pass_failure:
        lines.append("")
        lines.append("**Still failing tests (FIX INCOMPLETE):**")
        for test in result.fail_to_pass_failure[:10]:
            lines.append(f"  - {test}")
        if len(result.fail_to_pass_failure) > 10:
            lines.append(f"  ... and {len(result.fail_to_pass_failure) - 10} more failing tests")
    
    lines.append("")
    
    # PASS_TO_PASS analysis
    lines.extend([
        "## PASS_TO_PASS Tests (these should remain passing)",
        f"- Still passing: {len(result.pass_to_pass_success)}",
        f"- Now broken: {len(result.pass_to_pass_failure)}",
    ])
    
    if result.pass_to_pass_failure:
        lines.append("")
        lines.append("**Broken tests (REGRESSION INTRODUCED):**")
        for test in result.pass_to_pass_failure[:10]:
            lines.append(f"  - {test}")
        if len(result.pass_to_pass_failure) > 10:
            lines.append(f"  ... and {len(result.pass_to_pass_failure) - 10} more")
    
    lines.append("")
    
    # Diagnosis summary
    lines.append("## Diagnosis")
    if result.resolved:
        lines.append("The fix is correct and complete.")
    else:
        if result.fix_incomplete:
            lines.append("Fix is INCOMPLETE - some target tests still fail.")
        if result.has_regression:
            lines.append("REGRESSION - the fix broke previously passing tests.")
        if not result.fail_to_pass_success and not result.fail_to_pass_failure:
            lines.append("No test results - evaluation may have failed.")
        lines.append("")
        lines.append("**ACTION REQUIRED**: Read `test_output.txt` to see actual error messages and stack traces.")
    
    # Patch summary
    if result.patch:
        lines.extend([
            "",
            "## Patch Summary",
            f"Patch length: {len(result.patch)} characters",
        ])
        patch_lines = result.patch.split("\n")
        if len(patch_lines) > 30:
            lines.append("(truncated)")
            lines.extend(patch_lines[:30])
            lines.append("...")
        else:
            lines.extend(patch_lines)
    
    return "\n".join(lines)
