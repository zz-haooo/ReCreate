"""
Math Evaluator - Evaluation logic for mathematical reasoning tasks.

Core functions adapted from:
- evalchemy/eval/chat_benchmarks/LiveBench/livebench/process_results/math/AMPS_Hard/utils.py
- evalchemy/eval/chat_benchmarks/LiveBench/livebench/process_results/util.py

The evaluation uses SymPy to parse LaTeX expressions and determine mathematical equivalence.
"""

import re
import warnings
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional


@dataclass
class MathResult:
    """Math evaluation result."""
    instance_id: str
    passed: bool = False
    
    # Answers
    expected_answer: str = ""
    extracted_answer: str = ""
    
    # Metadata
    problem_type: str = ""   # algebra, geometry, number_theory, etc.
    source: str = ""         # MATH500, AIME24, etc.
    level: int = 0           # difficulty level (1-5 for MATH)
    
    # Execution details
    result: str = ""         # "passed", "failed", "timeout", "parse_error"
    error_message: str = ""
    
    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "passed": self.passed,
            "expected_answer": self.expected_answer,
            "extracted_answer": self.extracted_answer,
            "problem_type": self.problem_type,
            "source": self.source,
            "level": self.level,
            "result": self.result,
            "error_message": self.error_message,
        }


# ============================================================================
# Answer Extraction (from LiveBench util.py)
# ============================================================================

def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last \\boxed{...} or \\fbox{...} from a string."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1].replace("$", "").replace("fbox", "boxed")

    return retval


def remove_boxed(s: str) -> str:
    """Remove \\boxed{} wrapper from a string."""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def extract_boxed_answer(text: str) -> str:
    """Extract the answer from \\boxed{} in the text."""
    boxed = last_boxed_only_string(text)
    if boxed:
        try:
            return remove_boxed(boxed)
        except AssertionError:
            return ""
    return ""


# ============================================================================
# Answer Normalization (from AMPS_Hard utils.py)
# ============================================================================

def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.
    
    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    # Extract answer that is in LaTeX math, is bold, is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{\[])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def preprocess_answer(answer: str) -> str:
    """Preprocess answer string for comparison."""
    answer = answer.replace("+C", "")
    answer = answer.replace("+ C", "")
    answer = answer.replace("+ c", "")
    answer = answer.replace("+c", "")
    answer = answer.replace("\\\\fbox{", "\\\\boxed{")
    answer = answer.replace("\\dfrac", "\\frac")
    answer = answer.replace("\\tfrac", "\\frac")
    answer = answer.replace("\\left", "")
    answer = answer.replace("\\right", "")
    answer = answer.replace("\\bigl", "")
    answer = answer.replace("\\bigr", "")
    answer = answer.replace("\\Bigl", "")
    answer = answer.replace("\\Bigr", "")
    answer = answer.replace("\\,", "")
    answer = answer.replace("\\;", "")
    answer = answer.replace("\n", "")
    answer = answer.replace("\\cdot", "*")
    return answer


# ============================================================================
# Mathematical Equivalence (from AMPS_Hard utils.py)
# ============================================================================

def run_with_timeout(func, args=(), timeout=8):
    """Run a function with timeout protection."""
    def wrapper(queue):
        try:
            result = func(*args)
            queue.put(result)
        except Exception as e:
            queue.put(e)

    queue = Queue()
    process = Process(target=wrapper, args=(queue,))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError("Operation timed out")

    if queue.empty():
        raise TimeoutError("No result returned")
    
    result = queue.get()
    if isinstance(result, Exception):
        raise result
    return result


def parse_latex(x: str) -> list:
    """Parse LaTeX expression using SymPy."""
    try:
        import sympy
        from sympy.parsing.latex import parse_latex as sympy_parse_latex
    except ImportError:
        raise ImportError("sympy is required for math evaluation")
    
    try:
        import lark
    except ImportError:
        raise ImportError("lark is required for LaTeX parsing")
    
    try:
        # First try to parse with lark backend
        parsed_xs = sympy_parse_latex(x, backend="lark")
    except Exception:
        try:
            # Try with escaped backslashes fixed
            parsed_xs = sympy_parse_latex(x.replace("\\\\", "\\"), backend="lark")
        except Exception:
            try:
                # Fall back to default backend
                parsed_xs = sympy_parse_latex(x)
            except Exception:
                warnings.warn(f"couldn't parse {x}")
                return []

    if isinstance(parsed_xs, lark.Tree):
        # lark backend returns multiple options if there is ambiguity
        parsed_xs = parsed_xs.children
    else:
        parsed_xs = [parsed_xs]
    return parsed_xs


def is_equiv(x1: str, x2: str) -> bool:
    """
    Check if two mathematical expressions are equivalent.
    
    Uses SymPy to parse and simplify expressions.
    x1 and x2 are normalized latex strings.
    """
    try:
        import sympy
    except ImportError:
        raise ImportError("sympy is required for math evaluation")
    
    try:
        parsed_x1s = parse_latex(x1)
        parsed_x2s = parse_latex(x2)

        if len(parsed_x1s) == 0 or len(parsed_x2s) == 0:
            # Try direct string comparison as fallback
            return x1.strip() == x2.strip()

        errors = []
        for parsed_x1 in parsed_x1s:
            for parsed_x2 in parsed_x2s:
                try:
                    diff = parsed_x1 - parsed_x2
                except Exception as e:
                    errors.append(f"couldn't subtract {x1} and {x2}: {e}")
                    continue

                try:
                    if sympy.simplify(diff) == 0:
                        return True
                except Exception as e:
                    errors.append(f"couldn't compare simplified {x1} - {x2} with 0: {e}")

                try:
                    if sympy.Abs(sympy.simplify(diff)) < 0.001:
                        return True
                except Exception as e:
                    errors.append(f"Had some trouble simplifying when comparing {x1} and {x2}: {e}")
        
        for error in errors:
            warnings.warn(error)
        return False
    except ImportError as e:
        warnings.warn(str(e))
        raise
    except Exception as e:
        warnings.warn(f"Failed comparing {x1} and {x2}: {e}")
        traceback.print_tb(e.__traceback__)
        return False


def is_equiv_safe(expected: str, generated: str, timeout: float = 8) -> tuple[bool, str]:
    """
    Safe wrapper for is_equiv with timeout and preprocessing.
    
    Returns:
        (is_correct, result_message)
    """
    try:
        # Preprocess
        expected_clean = preprocess_answer(expected)
        generated_clean = preprocess_answer(generated)
        
        # Normalize
        expected_norm = normalize_final_answer(expected_clean)
        generated_norm = normalize_final_answer(generated_clean)
        
        # First try exact match
        if expected_norm.strip() == generated_norm.strip():
            return True, "passed (exact match)"
        
        # Try numeric comparison for simple numbers
        try:
            exp_num = float(expected_norm.replace(",", ""))
            gen_num = float(generated_norm.replace(",", ""))
            if abs(exp_num - gen_num) < 1e-6:
                return True, "passed (numeric match)"
        except (ValueError, TypeError):
            pass
        
        # Use SymPy equivalence with timeout
        result = run_with_timeout(is_equiv, (expected_norm, generated_norm), timeout=timeout)
        if result:
            return True, "passed (sympy equiv)"
        return False, "failed"
        
    except TimeoutError:
        return False, "timeout"
    except Exception as e:
        return False, f"error: {str(e)[:100]}"


# ============================================================================
# Main Evaluator Class
# ============================================================================

class MathEvaluator:
    """Evaluates mathematical reasoning tasks."""
    
    def __init__(self, timeout: float = 8):
        self.timeout = timeout
    
    def evaluate(
        self,
        instance_id: str,
        expected_answer: str,
        generated_text: str,
        problem_type: str = "",
        source: str = "",
        level: int = 0,
    ) -> MathResult:
        """
        Evaluate a math problem solution.
        
        Args:
            instance_id: Problem identifier
            expected_answer: Ground truth answer
            generated_text: Full generated text (may contain reasoning + \\boxed{answer})
            problem_type: Type of problem (algebra, geometry, etc.)
            source: Data source (MATH500, AIME24, etc.)
            level: Difficulty level
            
        Returns:
            MathResult with pass/fail status
        """
        # Extract answer from generated text
        extracted = extract_boxed_answer(generated_text)
        
        if not extracted:
            # Try to find answer at the end of text (fallback)
            lines = generated_text.strip().split('\n')
            if lines:
                extracted = lines[-1].strip()
        
        if not extracted:
            return MathResult(
                instance_id=instance_id,
                passed=False,
                expected_answer=expected_answer,
                extracted_answer="",
                problem_type=problem_type,
                source=source,
                level=level,
                result="no_answer",
                error_message="Could not extract answer from output",
            )
        
        # Check equivalence
        passed, result_msg = is_equiv_safe(expected_answer, extracted, self.timeout)
        
        return MathResult(
            instance_id=instance_id,
            passed=passed,
            expected_answer=expected_answer,
            extracted_answer=extracted,
            problem_type=problem_type,
            source=source,
            level=level,
            result=result_msg,
            error_message="" if passed else result_msg,
        )
    
    def evaluate_from_file(
        self,
        instance_id: str,
        expected_answer: str,
        solution_file: Path,
        problem_type: str = "",
        source: str = "",
        level: int = 0,
    ) -> MathResult:
        """Evaluate from a solution file."""
        if not solution_file.exists():
            return MathResult(
                instance_id=instance_id,
                passed=False,
                expected_answer=expected_answer,
                result="no_file",
                error_message=f"Solution file not found: {solution_file}",
            )
        
        generated_text = solution_file.read_text()
        return self.evaluate(
            instance_id=instance_id,
            expected_answer=expected_answer,
            generated_text=generated_text,
            problem_type=problem_type,
            source=source,
            level=level,
        )


def format_math_result_for_recreate_agent(result: MathResult) -> str:
    """Format MathResult for ReCreate-Agent consumption."""
    lines = [
        "=" * 60,
        "MATH EVALUATION RESULT",
        "=" * 60,
        "",
        f"**Instance**: {result.instance_id}",
        f"**Source**: {result.source}",
        f"**Type**: {result.problem_type}",
        f"**Level**: {result.level}",
        f"**Passed**: {'✓ YES' if result.passed else '✗ NO'}",
        "",
    ]
    
    if not result.passed:
        lines.extend([
            "## Error Details",
            f"Result: {result.result}",
            "",
            "## Answers",
            f"Expected: {result.expected_answer}",
            f"Extracted: {result.extracted_answer if result.extracted_answer else '(none)'}",
            "",
            "## Analysis Hints",
            "- Check if the answer uses \\boxed{} format",
            "- Verify numerical precision and format",
            "- Check for equivalent mathematical forms",
            "- Ensure symbolic expressions are simplified correctly",
        ])
    else:
        lines.extend([
            "## Success",
            f"Expected: {result.expected_answer}",
            f"Extracted: {result.extracted_answer}",
            "The solution is mathematically correct.",
        ])
    
    return "\n".join(lines)


# Export key classes and functions
__all__ = [
    "MathResult",
    "MathEvaluator",
    "extract_boxed_answer",
    "normalize_final_answer",
    "is_equiv",
    "is_equiv_safe",
    "format_math_result_for_recreate_agent",
]

