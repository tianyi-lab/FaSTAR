from __future__ import annotations
import json, time, uuid, os
import subprocess
from pathlib import Path
from PIL import Image

TRACE_BUFFER_PATH = Path("trace_buffer.jsonl")
SUBROUTINE_RULE_TABLE_PATH = Path("subroutine_rule_table.json")
REFINE_FREQ_K = 20

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def _write_json(path: Path, obj): path.write_text(json.dumps(obj, indent=2))

def _load_rules() -> dict:  
    return json.loads(SUBROUTINE_RULE_TABLE_PATH.read_text()) if SUBROUTINE_RULE_TABLE_PATH.exists() else {}

def _json_safe(obj):
    if isinstance(obj, Image.Image):
        return None
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items() if not isinstance(v, Image.Image)}
    return obj

def _count_traces() -> int:
    if not TRACE_BUFFER_PATH.exists():
        return 0
    try:
        with open(TRACE_BUFFER_PATH, 'r') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def append_trace(trace: dict) -> None:

    required_fields = ["subtask", "path", "tools_executed", "success", "cost", "quality"]
    for field in required_fields:
        if field not in trace:
            print(f"Warning: Trace missing required field '{field}'. Adding default value.")
            if field in ["success", "cost", "quality"]:
                trace[field] = {"success": False, "cost": 0, "quality": 0}[field]
            else:
                trace[field] = []

    if "tool_outputs" in trace:
        for tool, outputs in trace["tool_outputs"].items():
            if isinstance(outputs, dict) and "image" in outputs:
                del outputs["image"]
    else:
        trace["tool_outputs"] = {}

    if "context_features" not in trace:
        trace["context_features"] = {}
    
    with TRACE_BUFFER_PATH.open("a") as f:
        f.write(json.dumps(_json_safe(trace)) + "\n")

def refine_rules_if_needed(llm_key: str) -> bool:
    
    trace_count = _count_traces()
    if trace_count < REFINE_FREQ_K:
        return False
    
    print(f"Triggering inductive reasoning after {trace_count} traces")
    
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = llm_key
    
    try:
        result = subprocess.run(
            ["python", "run1.py"],
            env=env,
            check=True,
            capture_output=True,
            text=True
        )

        print("Run1 script output: ", result.stdout)
        
        if "Updated subroutine rule table" in result.stdout:
            print("Inductive reasoning completed successfully. Subroutine rule table updated.")
            return True
        else:
            print("Inductive reasoning completed, but no updates were made.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error running inductive reasoning: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error running inductive reasoning: {e}")
        return False
