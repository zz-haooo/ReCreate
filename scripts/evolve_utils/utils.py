"""Common utilities for parallel evolution."""

import logging
import re
import yaml
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from rich.console import Console

console = Console()
console_lock = Lock()


def safe_print(msg: str):
    """Thread-safe print."""
    with console_lock:
        console.print(msg)


def clean_yaml_content(content: str) -> str:
    """Clean illegal control characters from YAML content.
    
    Some ReCreate-Agent outputs may contain control characters (like backspace \x08)
    that cause YAML parsing to fail. This function removes them.
    """
    # Remove control characters 0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F (keep \t, \n, \r)
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)


def safe_load_yaml(file_path: Path) -> dict:
    """Safely load YAML file, cleaning illegal characters if needed."""
    if not file_path.exists():
        return {}
    try:
        content = clean_yaml_content(file_path.read_text())
        # Try to load YAML
        result = yaml.safe_load(content)
        if result is None:
            return {}
        return result
    except yaml.YAMLError as e:
        error_msg = str(e)
        logging.warning(f"YAML parsing error in {file_path}: {error_msg}")
        # Try to fix common issues: heredoc end markers that look like YAML keys
        # If error mentions "could not find expected ':'" and involves heredoc markers
        if "could not find expected ':'" in error_msg and ("PY" in error_msg or "EOF" in error_msg or "END" in error_msg):
            try:
                lines = content.split('\n')
                fixed_lines = []
                in_multiline_string = False
                multiline_indent = 0
                
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    # Detect start of multi-line string
                    if ':' in line and ('|' in line or '>' in line):
                        in_multiline_string = True
                        multiline_indent = len(line) - len(line.lstrip())
                        fixed_lines.append(line)
                    # Detect end of multi-line string (next key at same or less indent)
                    elif in_multiline_string and stripped and not stripped.startswith('#'):
                        current_indent = len(line) - len(line.lstrip())
                        # If this looks like a YAML key (has colon and proper indent)
                        if ':' in line and current_indent <= multiline_indent:
                            in_multiline_string = False
                        # If this is a heredoc end marker at the start of line, ensure it's indented
                        elif stripped in ['PY', 'EOF', 'END'] and current_indent <= multiline_indent:
                            # Indent it to match the multiline string content
                            fixed_lines.append(' ' * (multiline_indent + 2) + stripped)
                            continue
                    fixed_lines.append(line)
                
                fixed_content = '\n'.join(fixed_lines)
                result = yaml.safe_load(fixed_content)
                if result is None:
                    return {}
                return result
            except Exception:
                return {}
        return {}
    except Exception as e:
        logging.warning(f"Unexpected error loading YAML from {file_path}: {e}")
        return {}


def chunks(lst: list, n: int):
    """Split list into chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


@dataclass
class BatchResult:
    """Result of a single instance evolution in a batch."""
    instance_id: str
    success: bool
    score: float
    scaffold_changed: bool
    has_new_tools: bool
    has_new_memories: bool
    error: str = ""
    exit_status: str = ""
    duration: float = 0.0
