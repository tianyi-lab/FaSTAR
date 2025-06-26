import os
import json
import random
import datetime
import difflib
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from openai import OpenAI
from PIL import Image

from run import main as run_main

TRACE_BUFFER_PATH = "trace_buffer.jsonl"
SUBROUTINE_RULE_TABLE_PATH = "subroutine_rule_table.json"
DATASET_PATH = "dataset"
TEST_SAMPLES_COUNT = 25
MAX_RETRIES = 2


class InductiveReasoner:
    def __init__(self, llm_api_key: str):
        self.llm_api_key = llm_api_key
        self.client = OpenAI(api_key=llm_api_key)

    def _load_traces(self) -> List[Dict]:
        traces = []
        try:
            with open(TRACE_BUFFER_PATH, "r") as f:
                for line in f:
                    if line.strip():
                        traces.append(json.loads(line))
            return traces
        except FileNotFoundError:
            print(f"Trace buffer not found at {TRACE_BUFFER_PATH}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing trace buffer: {e}")
            return []

    def _load_subroutine_table(self) -> Dict:
        try:
            with open(SUBROUTINE_RULE_TABLE_PATH, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(
                f"Warning: Could not load subroutine rule table from {SUBROUTINE_RULE_TABLE_PATH}. Using empty table."
            )
            return {}

    def _save_subroutine_table(self, table: Dict) -> None:
        with open(SUBROUTINE_RULE_TABLE_PATH, "w") as f:
            json.dump(table, f, indent=2)
        print(f"Updated subroutine rule table saved to {SUBROUTINE_RULE_TABLE_PATH}")

    def _clear_trace_buffer(self) -> None:
        try:
            os.remove(TRACE_BUFFER_PATH)
            print(f"Trace buffer cleared: {TRACE_BUFFER_PATH}")
        except FileNotFoundError:
            pass

    def _generate_subroutine_proposals(
        self, traces: List[Dict], current_table: Dict
    ) -> Dict:
        system_prompt = """
        Goal: Analyze the provided experimental run data for specific tasks (e.g., Object Recoloration) 
        to infer initial, potentially qualitative, activation rules (preference conditions) for
        each distinct execution path employed.

        So we run different models and tools for different or same image editing tasks and store the
        observations including what path was finally used and what were the conditions of objects, etc.
        and this data is provided to you. Now we wish to infer some subroutines or commonly used
        paths and their activation rules under which they are commonly activated. Can you find some
        commonly used subroutines or paths and infer some rules for these paths using the status
        of these cases and other factors and give the rules for both paths and they need not be too
        specific but a bit vague is fine like if you observe that some particular path always fails in case
        object size is less then you can give the rule that this path should be used when object is not
        too small and not give any specific values so activation rule will include like object_size
        = not too small, etc like this based on all factors like object size, color transitions, etc
        and also it is possible that for some path it failed bcz of some specific condition like its not
        necessary all conditions led to failure so you need to check which is the condition which
        always leads to failures or which always leads to success and that will constitute a rule if
        some condition leads to both failures and success with same value then it means that this is
        not the contributing factor and there's something else that's causing the failure or success and
        keep in mind that output rules should be of activation format like in what cases this should be
        used and not negative ones so if there is some path which always fails when object size is
        big then your activation rule will have object_size = small and not some deactivation
        rules which has object_size = big. You should also include some explanatory examples
        in the rule which can help some person or LLM understand them better when referring to
        these rules. eg. if there is a rule where you want to say that this path will only succeed
        when the difference between size of objects is not too big then you can have a rule like
        : "size_difference(original, target objects) = Not too big (eg. hen to
        car, etc)" where you include some example.

        You should focus on activation rules which are like in what case this particular path will always succeed 
        and some activation rules should also include a kind of deactivation rule with a not like in case you 
        observe that some path always fails when there is some condition x where x can be like object is too 
        small or color difference is huge then you should infer an activation rule that is negate of this like the rule
        can be object is "not" too small or color difference is "not" huge so that these activation rules
        can act as a kind of deactivation rules as well and prevent the path from getting activated in
        cases where we know for sure it'll fail.

        The output format for each path for which you can infer some rule/s will be following:
        {
          "add": [
            {
              "subtask": "Object Recoloration",
              "tools": ["Grounding DINO", "SAM", "SD Inpaint"],
              "activation_rules": [
                "object_size: Not Too Small",
                "overlapping_critical_elements: None (eg. Some text written on object to be recolored and this text is critical for some future or past subtask)"
              ],
              "C": 10.39,
              "Q": 0.89
            },
            ...
          ],
          "update": [
            {
              "id": "SR1",
              "activation_rules": [
                "object_size: Not Too Small",
                "overlapping_critical_elements: None (eg. Some text written on object to be recolored and this text is critical for some future or past subtask)",
                "NEW RULE HERE"
              ]
            },
            ...
          ]
        }
        
        Return a JSON object with two keys:
        - "add": List of new subroutines to add
        - "update": List of existing subroutines to update
        
        For each new subroutine, include:
        - "subtask": The subtask type
        - "tools": List of tools in the subroutine
        - "activation_rules": List of rules for when to use this subroutine
        - "C": Average cost observed in the traces
        - "Q": Average quality observed in the traces
        
        For each update, include:
        - "id": The ID of the existing subroutine to update
        - "activation_rules": The updated list of activation rules
        """

        user_prompt = (
            "CURRENT_RULE_TABLE:\n"
            + json.dumps(current_table, indent=2)
            + "\n\nRECENT_TRACES:\n"
            + json.dumps(traces, indent=2)
            + "\n\nAnalyze these traces and the current rule table to propose new subroutines or updates to existing ones."
        )

        try:
            response = self.client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            content = response.choices[0].message.content
            try:
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_str = content.split("```")[1].strip()
                else:
                    json_str = content.strip()

                proposals = json.loads(json_str)
                return proposals
            except (json.JSONDecodeError, IndexError) as e:
                print(f"Error parsing LLM response: {e}")
                print(f"Raw response: {content}")
                return {"add": [], "update": []}

        except Exception as e:
            print(f"Error calling LLM for subroutine proposals: {e}")
            return {"add": [], "update": []}

    def _get_test_samples(
        self, subtask: str, count: int = TEST_SAMPLES_COUNT
    ) -> List[Tuple[str, str]]:
        folder_name = subtask.lower().replace(" ", "_")
        folder_path = Path(DATASET_PATH) / folder_name

        if not folder_path.exists():
            print(
                f"Warning: Dataset folder not found for subtask '{subtask}' at {folder_path}"
            )
            return []

        samples = []
        for img_path in folder_path.glob("*.png"):
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                try:
                    with open(txt_path, "r") as f:
                        prompt = f.read().strip()
                    samples.append((str(img_path), prompt))
                except Exception as e:
                    print(f"Error reading sample {img_path}: {e}")

        if len(samples) < count:
            print(
                f"Warning: Only {len(samples)} samples found for subtask '{subtask}', requested {count}"
            )
            return samples
        return random.sample(samples, count)

    def _test_subroutine(
        self,
        subtask: str,
        samples: List[Tuple[str, str]],
        original_table: Dict,
        modified_table: Dict,
    ) -> Tuple[Dict, Dict]:
        temp_original = SUBROUTINE_RULE_TABLE_PATH + ".original"
        temp_modified = SUBROUTINE_RULE_TABLE_PATH + ".modified"

        with open(temp_original, "w") as f:
            json.dump(original_table, f, indent=2)

        with open(temp_modified, "w") as f:
            json.dump(modified_table, f, indent=2)

        print(f"Testing with original table on {len(samples)} samples for subtask '{subtask}'...")
        original_results = {"cost": 0, "quality": 0, "traces": []}

        os.rename(temp_original, SUBROUTINE_RULE_TABLE_PATH)

        for i, (image_path, prompt) in enumerate(samples):
            print(f"  Sample {i + 1}/{len(samples)}: {image_path}")
            try:
                if os.path.exists(TRACE_BUFFER_PATH):
                    os.remove(TRACE_BUFFER_PATH)

                _, _, cost, quality = run_main(
                    image_path,
                    prompt,
                    output_chain=f"temp_chain_{i}.json",
                    output_image=f"temp_output_{i}.png",
                )

                original_results["cost"] += cost
                original_results["quality"] += quality

                if os.path.exists(TRACE_BUFFER_PATH):
                    with open(TRACE_BUFFER_PATH, "r") as f:
                        for line in f:
                            if line.strip():
                                original_results["traces"].append(json.loads(line))
            except Exception as e:
                print(f"  Error testing sample {i + 1}: {e}")

        if samples:
            original_results["cost"] /= len(samples)
            original_results["quality"] /= len(samples)

        print(f"Testing with modified table on {len(samples)} samples for subtask '{subtask}'...")
        modified_results = {"cost": 0, "quality": 0, "traces": []}

        os.rename(temp_modified, SUBROUTINE_RULE_TABLE_PATH)

        for i, (image_path, prompt) in enumerate(samples):
            print(f"  Sample {i + 1}/{len(samples)}: {image_path}")
            try:
                if os.path.exists(TRACE_BUFFER_PATH):
                    os.remove(TRACE_BUFFER_PATH)

                _, _, cost, quality = run_main(
                    image_path,
                    prompt,
                    output_chain=f"temp_chain_{i}.json",
                    output_image=f"temp_output_{i}.png",
                )

                modified_results["cost"] += cost
                modified_results["quality"] += quality

                if os.path.exists(TRACE_BUFFER_PATH):
                    with open(TRACE_BUFFER_PATH, "r") as f:
                        for line in f:
                            if line.strip():
                                modified_results["traces"].append(json.loads(line))
            except Exception as e:
                print(f"  Error testing sample {i + 1}: {e}")

        if samples:
            modified_results["cost"] /= len(samples)
            modified_results["quality"] /= len(samples)

        for i in range(len(samples)):
            for file in [f"temp_chain_{i}.json", f"temp_output_{i}.png"]:
                if os.path.exists(file):
                    os.remove(file)

        with open(SUBROUTINE_RULE_TABLE_PATH, "w") as f:
            json.dump(original_table, f, indent=2)

        return original_results, modified_results

    def _calculate_net_benefit(
        self, original_results: Dict, modified_results: Dict
    ) -> float:
        cost_change = (
            (modified_results["cost"] - original_results["cost"])
            / original_results["cost"]
            * 100
        )
        quality_change = (
            (modified_results["quality"] - original_results["quality"])
            / original_results["quality"]
            * 100
        )
        net_benefit = cost_change - quality_change
        print(
            f"Original: Cost={original_results['cost']:.2f}, Quality={original_results['quality']:.2f}"
        )
        print(
            f"Modified: Cost={modified_results['cost']:.2f}, Quality={modified_results['quality']:.2f}"
        )
        print(f"Changes: Cost={cost_change:.2f}%, Quality={quality_change:.2f}%")
        print(f"Net Benefit: {net_benefit:.2f} (negative is better)")
        return net_benefit

    def _refine_proposal(
        self, proposal: Dict, original_results: Dict, modified_results: Dict
    ) -> Dict:
        system_prompt = """
        You are tasked with refining a proposed subroutine rule that did not pass the net benefit test.
        The goal is to modify the activation rules to improve the cost-quality trade-off.
        
        Analyze the test results from both the original and modified subroutine tables, and suggest
        improvements to the activation rules that would:
        1. Reduce cost further without sacrificing quality
        2. Improve quality without increasing cost
        3. Find a better balance between cost and quality
        
        Focus on making the activation rules more precise and targeted to the specific conditions
        where the subroutine performs best.
        """

        user_prompt = (
            "ORIGINAL PROPOSAL:\n"
            + json.dumps(proposal, indent=2)
            + "\n\nORIGINAL TEST RESULTS:\n"
            + json.dumps(original_results, indent=2)
            + "\n\nMODIFIED TEST RESULTS:\n"
            + json.dumps(modified_results, indent=2)
            + "\n\nThe proposal did not pass the net benefit test. Please refine the activation rules to improve the cost-quality trade-off."
        )

        try:
            response = self.client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            content = response.choices[0].message.content
            try:
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_str = content.split("```")[1].strip()
                else:
                    json_str = content.strip()
                refined_proposal = json.loads(json_str)
                return refined_proposal
            except (json.JSONDecodeError, IndexError) as e:
                print(f"Error parsing LLM response for refinement: {e}")
                print(f"Raw response: {content}")
                return proposal
        except Exception as e:
            print(f"Error calling LLM for proposal refinement: {e}")
            return proposal

    def process_traces(self) -> None:
        traces = self._load_traces()
        if not traces:
            print("No traces found. Exiting.")
            return
        print(f"Loaded {len(traces)} traces from {TRACE_BUFFER_PATH}")
        current_table = self._load_subroutine_table()
        print(f"Loaded subroutine table with {len(current_table)} entries")
        proposals = self._generate_subroutine_proposals(traces, current_table)
        for add_proposal in proposals.get("add", []):
            subtask = add_proposal.get("subtask")
            tools = add_proposal.get("tools", [])
            if not subtask or not tools:
                print("Invalid add proposal: missing subtask or tools")
                continue
            print(f"\nProcessing add proposal for subtask '{subtask}' with tools {tools}")
            exists = False
            for sr_id, sr_info in current_table.items():
                if (
                    sr_info.get("subtask") == subtask
                    and sr_info.get("tools") == tools
                ):
                    exists = True
                    print(f"Subroutine already exists as {sr_id}")
                    break
            if exists:
                continue
            modified_table = current_table.copy()
            new_id = f"SR{len(current_table) + 1}"
            modified_table[new_id] = {
                "subtask": subtask,
                "tools": tools,
                "activation_rules": add_proposal.get("activation_rules", []),
                "avg_cost": float(add_proposal.get("C", 0)),
                "avg_quality": float(add_proposal.get("Q", 1)),
            }
            samples = self._get_test_samples(subtask)
            if not samples:
                print(f"No test samples found for subtask '{subtask}'")
                continue
            accepted = False
            for retry in range(MAX_RETRIES + 1):
                print(f"\nTesting add proposal (attempt {retry + 1}/{MAX_RETRIES + 1})...")
                original_results, modified_results = self._test_subroutine(
                    subtask, samples, current_table, modified_table
                )
                net_benefit = self._calculate_net_benefit(
                    original_results, modified_results
                )
                if net_benefit < 0:
                    print(f"Add proposal passed with net benefit {net_benefit:.2f}")
                    current_table[new_id] = modified_table[new_id]
                    accepted = True
                    break
                elif retry < MAX_RETRIES:
                    print(
                        f"Add proposal failed with net benefit {net_benefit:.2f}. Refining..."
                    )
                    refined = self._refine_proposal(
                        add_proposal, original_results, modified_results
                    )
                    modified_table[new_id]["activation_rules"] = refined.get(
                        "activation_rules",
                        add_proposal.get("activation_rules", []),
                    )
                    samples = self._get_test_samples(subtask)
                else:
                    print(f"Add proposal failed after {MAX_RETRIES + 1} attempts")
            if accepted:
                print(f"Added new subroutine {new_id} for subtask '{subtask}'")
        for update_proposal in proposals.get("update", []):
            sr_id = update_proposal.get("id")
            if not sr_id or sr_id not in current_table:
                print(
                    f"Invalid update proposal: {sr_id} not found in current table"
                )
                continue
            subtask = current_table[sr_id].get("subtask")
            print(
                f"\nProcessing update proposal for {sr_id} (subtask '{subtask}')"
            )
            modified_table = current_table.copy()
            modified_table[sr_id] = current_table[sr_id].copy()
            modified_table[sr_id]["activation_rules"] = update_proposal.get(
                "activation_rules",
                current_table[sr_id].get("activation_rules", []),
            )
            samples = self._get_test_samples(subtask)
            if not samples:
                print(f"No test samples found for subtask '{subtask}'")
                continue
            accepted = False
            for retry in range(MAX_RETRIES + 1):
                print(
                    f"\nTesting update proposal (attempt {retry + 1}/{MAX_RETRIES + 1})..."
                )
                original_results, modified_results = self._test_subroutine(
                    subtask, samples, current_table, modified_table
                )
                net_benefit = self._calculate_net_benefit(
                    original_results, modified_results
                )
                if net_benefit < 0:
                    print(
                        f"Update proposal passed with net benefit {net_benefit:.2f}"
                    )
                    current_table[sr_id] = modified_table[sr_id]
                    accepted = True
                    break
                elif retry < MAX_RETRIES:
                    print(
                        f"Update proposal failed with net benefit {net_benefit:.2f}. Refining..."
                    )
                    refined = self._refine_proposal(
                        update_proposal, original_results, modified_results
                    )
                    modified_table[sr_id]["activation_rules"] = refined.get(
                        "activation_rules",
                        update_proposal.get("activation_rules", []),
                    )
                    samples = self._get_test_samples(subtask)
                else:
                    print(f"Update proposal failed after {MAX_RETRIES + 1} attempts")
            if accepted:
                print(f"Updated subroutine {sr_id} for subtask '{subtask}'")
        self._save_subroutine_table(current_table)


def main():
    parser = argparse.ArgumentParser(
        description="Run inductive reasoning to update subroutine rule table."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force processing even if trace count is below threshold",
    )

    args = parser.parse_args()

    try:
        trace_count = sum(1 for _ in open(TRACE_BUFFER_PATH))
        if trace_count < 20 and not args.force:
            print(f"Not enough traces ({trace_count}/20). Use --force to process anyway.")
            return
    except FileNotFoundError:
        print(f"Trace buffer not found at {TRACE_BUFFER_PATH}")
        return

    llm_api_key = os.getenv("OPENAI_API_KEY")
    if not llm_api_key:
        print("Warning: OPENAI_API_KEY not set. Using empty string.")
        llm_api_key = ""

    reasoner = InductiveReasoner(llm_api_key)
    reasoner.process_traces()


if __name__ == "__main__":
    main()
