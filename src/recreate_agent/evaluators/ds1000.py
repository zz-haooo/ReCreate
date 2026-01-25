"""
DS-1000 Evaluator

Evaluates data science code completion tasks using official DS-1000 evaluation.
Returns pass/fail result similar to SWE-bench.

Based on: https://github.com/xlang-ai/DS-1000
"""

import contextlib
import io
import multiprocessing
import os
import signal
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DS1000Result:
    """DS-1000 evaluation result."""
    instance_id: str
    passed: bool = False
    
    # Metadata
    library: str = ""           # Pandas, Numpy, Matplotlib, etc.
    perturbation_type: str = "" # Origin, Semantic, etc.
    
    # Execution details
    result: str = ""            # "passed", "failed: ...", "timed out"
    error_message: str = ""
    
    # Generated code
    generated_code: str = ""
    
    # Test details
    test_program: str = ""
    
    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "passed": self.passed,
            "library": self.library,
            "perturbation_type": self.perturbation_type,
            "result": self.result,
            "error_message": self.error_message,
        }


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when read."""
    def read(self, *args, **kwargs):
        raise IOError
    def readline(self, *args, **kwargs):
        raise IOError
    def readlines(self, *args, **kwargs):
        raise IOError
    def readable(self, *args, **kwargs):
        return False


class _redirect_stdin(contextlib._RedirectStream):
    _stream = 'stdin'


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with _redirect_stdin(stream):
                yield


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


def check_correctness(program: str, timeout: float = 120) -> dict:
    """
    Evaluates the correctness of generated code by running the test program.
    
    This is adapted from the official DS-1000 execution.py
    """
    def unsafe_execute():
        with create_tempdir():
            import os
            import shutil
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir_fn = os.chdir

            try:
                exec_globals = {}
                with swallow_io():
                    with time_limit(timeout):
                        exec(program, exec_globals)
                result.append("passed")
            except TimeoutException:
                result.append("timed out")
            except BaseException as e:
                result.append(f"failed: {e}")

            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir_fn

    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute)
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return {
        "passed": result[0] == "passed",
        "result": result[0],
    }


def postprocess_code(code: str) -> str:
    """
    Clean up generated code from common model output artifacts.
    Based on official DS-1000 postprocessing.
    """
    if isinstance(code, list):
        code = code[0]
    code = code.split('</code>')[0]
    code = code.replace('```python', '')
    code = code.split('```')[0]
    code = code.split('\nEND SOLUTION')[0]
    code = code.replace('<code>', '')
    return code.strip()


class DS1000Evaluator:
    """Evaluates DS-1000 code completion tasks."""
    
    def __init__(self, data_path: Path | str = "datasets/ds1000/data/ds1000.jsonl.gz"):
        self.data_path = Path(data_path)
        self._problems = None
    
    @property
    def problems(self) -> list[dict]:
        """Lazy load DS-1000 problems."""
        if self._problems is None:
            import gzip
            import json
            self._problems = [
                json.loads(l) 
                for l in gzip.open(self.data_path, "rt").readlines()
            ]
        return self._problems
    
    def get_problem(self, problem_id: int) -> dict:
        """Get problem by ID."""
        return self.problems[problem_id]
    
    def evaluate(
        self,
        problem_id: int,
        generated_code: str,
        timeout: float = 120,
    ) -> DS1000Result:
        """
        Evaluate generated code for a DS-1000 problem.
        
        Args:
            problem_id: Problem ID (0-999)
            generated_code: Generated code to test
            timeout: Execution timeout in seconds
            
        Returns:
            DS1000Result with pass/fail status
        """
        problem = self.get_problem(problem_id)
        
        # Clean up generated code
        code = postprocess_code(generated_code)
        
        # Build test program (from official test_ds1000.py)
        test_program = (
            problem['code_context'] + '\n'
            + f'code = {repr(code)}\n'
            + 'test_execution(code)\n'
            + ('test_string(code)\n' if 'test_string(' in problem['code_context'] else '\n')
        )
        
        # Execute test
        exec_result = check_correctness(test_program, timeout=timeout)
        
        return DS1000Result(
            instance_id=f"ds1000_{problem_id:04d}",
            passed=exec_result["passed"],
            library=problem['metadata']['library'],
            perturbation_type=problem['metadata']['perturbation_type'],
            result=exec_result["result"],
            error_message=exec_result["result"] if not exec_result["passed"] else "",
            generated_code=code,
            test_program=test_program,
        )
    
    def evaluate_batch(
        self,
        answers: list[str],
        max_workers: int = 16,
        timeout: float = 120,
    ) -> list[DS1000Result]:
        """
        Evaluate a batch of answers.
        
        Args:
            answers: List of generated code, indexed by problem_id
            max_workers: Number of parallel workers
            timeout: Execution timeout per problem
        """
        import concurrent.futures as cfuts
        from tqdm import tqdm
        
        results = []
        with cfuts.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futs = []
            for problem_id, code in enumerate(answers):
                futs.append(
                    executor.submit(self.evaluate, problem_id, code, timeout)
                )
            
            for f in tqdm(cfuts.as_completed(futs), total=len(futs)):
                results.append(f.result())
        
        return sorted(results, key=lambda r: r.instance_id)


def format_ds1000_result_for_recreate_agent(result: DS1000Result) -> str:
    """Format DS1000 result for ReCreate-Agent consumption."""
    lines = [
        "=" * 60,
        "DS-1000 EVALUATION RESULT",
        "=" * 60,
        "",
        f"**Instance**: {result.instance_id}",
        f"**Library**: {result.library}",
        f"**Perturbation**: {result.perturbation_type}",
        f"**Passed**: {'✓ YES' if result.passed else '✗ NO'}",
        "",
    ]
    
    if not result.passed:
        lines.extend([
            "## Error Details",
            f"Result: {result.result}",
            "",
        ])
        
        if result.error_message:
            lines.extend([
                "## Error Message",
                result.error_message[:500],
                "",
            ])
        
        lines.extend([
            "## Generated Code",
            "```python",
            result.generated_code[:1000] if result.generated_code else "(no code)",
            "```",
            "",
            "## Analysis Hints",
            "- Check if the code produces the expected output type",
            "- Verify variable names match what the test expects (often 'result')",
            "- For numerical results, check for floating point precision issues",
            "- For DataFrame operations, ensure column names and dtypes are correct",
        ])
    else:
        lines.extend([
            "## Success",
            "The generated code passed all test cases.",
        ])
    
    return "\n".join(lines)


# Export key classes and functions for use in domain_adapter
__all__ = [
    "DS1000Result",
    "DS1000Evaluator", 
    "postprocess_code",
    "format_ds1000_result_for_recreate_agent",
]

