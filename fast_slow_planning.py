import os
import json
import time
import yaml
import re
import base64
from PIL import Image
from openai import OpenAI
from collections import defaultdict
from pathlib import Path
import io

from subtask_chain import generate_subtask_chain
from tool_subgraph import build_tool_subgraph_from_subtask_chain
from astar_search import a_star_search, extract_metadata_from_node, compute_quality_dynamic
from main import ToolPipeline
from subroutine import generate_subroutine
from inductive_reasoning import append_trace, refine_rules_if_needed

SUBROUTINE_RULE_TABLE_PATH = "subroutine_rule_table.json"
TRACE_BUFFER_PATH = "trace_buffer.jsonl"
QUALITY_THRESHOLD = 0.8
TRACE_TRIGGER_COUNT = 20
TRACE_REQUIREMENTS_PATH = "configs/trace_requirements.yaml"

class FastSlowPlanner:
    def __init__(self, llm_api_key, alpha=0, quality_threshold=QUALITY_THRESHOLD):
        self._IMAGE_EDIT_KEYWORDS = (
            "Outpaint", "Inpaint", "Recolor", "RemoveBG", "Replace",
            "Colorize", "Expand", "Erase", "Restore"
        )
        self.llm_api_key = llm_api_key
        self.alpha = alpha
        self.quality_threshold = quality_threshold
        self.client = OpenAI(api_key=llm_api_key)
        self.pipeline = ToolPipeline("configs/tools.yaml", auto_install=True)
        self.subroutine_table = self._load_subroutine_table()
        self.trace_count = self._get_trace_count()
        self.trace_requirements = self._load_trace_requirements()

    def _is_image_editing_tool(self, tool_name: str) -> bool:
        return any(k in tool_name for k in self._IMAGE_EDIT_KEYWORDS)

    def _load_subroutine_table(self):
        try:
            with open(SUBROUTINE_RULE_TABLE_PATH, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Warning: Could not load subroutine rule table from {SUBROUTINE_RULE_TABLE_PATH}. Using empty table.")
            return {}

    def _load_trace_requirements(self):
        try:
            with open(TRACE_REQUIREMENTS_PATH, 'r') as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError):
            print(f"Warning: Could not load trace requirements from {TRACE_REQUIREMENTS_PATH}. Trace filtering will be disabled.")
            return {}

    def _get_trace_count(self):
        try:
            with open(TRACE_BUFFER_PATH, 'r') as f:
                return sum(1 for _ in f)
        except FileNotFoundError:
            return 0

    def _extract_context_features(self, image_path, subtask):
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        mime_type = "image/jpeg"
        if image_path.lower().endswith(".png"):
            mime_type = "image/png"

        metadata = extract_metadata_from_node(subtask)
        subtask_name = metadata["subtask_name"]
        from_object = metadata["from_object"]
        target_object = metadata["target_object"]

        prompt = f"""
        Analyze this image and extract the following context features for the {subtask_name} task
        involving {from_object} {f'and {target_object}' if target_object else ''}.

        Please provide these specific features in a JSON format:

        1. overlapping_critical_elements: So you need to see if there are any elements on the object which are important like some other object overlapping with current object specially something which might need to be edited in later subtasks like some text written on top of current object and this text needs to be removed/replaced/etc in later subtasks - Return True/False only.
        2. object_clarity: [Low/Medium/High] - How clearly visible and unambiguous the object is and is opaque, or if it is clearly visible or not. Return Low/Medium/High only. Single option possible.
        3. background_content_type: It describes the background behind current object if it is a simple plain background or some pattern which is either simple repeating or complex or if there are specific objects behind it which are currently occluded by current object or things like that. Multiple Options possible. Choose from-[Simple_Texture/Homogenous_Area/Repeating_Pattern/Complex_Scene/Occludes_Specific_Objects]
        4. background_reconstruction_need: If the background behind current object if removed needs to be simply filled, inpainted or needs complex drawing for maybe some objects which were occluded by current object which need to be redrawn or needs some semantic completion, etc. Multiple Options Possible. Answer from - [Filling/Inpainting/Drawing/Semantic_Completion/None]
        5. background_content_behind_text: So you need to describe the background behind the current text written which is to be replaced or removed and choose from following options- [Plain_Color/Simple_Gradient/Simple_Texture/Complex_Image/Uniform_Solid_Color]. Multiple options can be selected.
        6. surrounding_context_similarity: It is Low if the nearby areas do not have any text or some specific fine patterns which can affect current task like if it is text removal or replacement then nearby text which is not relevant to current subtask can affect current subtask also so in that case its High [High/Medium/Low] - Choose 1 option
        7. background_artifact_tolerance: If the artifacts behind current object or text are tolerant to changes like simple textures or simple objects like clouds which don't have any particular shape and are easy to recreate and can tolerate minor flaws then High else Medium or Low [High/Medium/Low] - choose 1 option


        Return only the JSON object with these features.
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ]
            }
        ]

        try:
            response = self.client.chat.completions.create(model="o4-mini", messages=messages)
            content = response.choices[0].message.content
            try:
                if '```json' in content:
                    json_str = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    json_str = content.split('```')[1].strip()
                else:
                    json_str = content.strip()
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError) as e:
                print(f"Error parsing LLM response: {e}\nRaw response: {content}")
                return {}
        except Exception as e:
            print(f"Error calling LLM for context extraction: {e}")
            return {}

    def _select_subroutine(self, subtask, subroutine_chain, context_features):
        subroutines = subroutine_chain.get(subtask, [])
        if subroutines and subroutines[0] != "None":
            if len(subroutines) == 1:
                sr_id = subroutines[0]
                return sr_id, self.subroutine_table[sr_id]["tools"]
            else:
                sub_cost = []
                for sub in subroutines:
                    if sub in self.subroutine_table:
                        time = self.subroutine_table[sub]["avg_cost"]
                        quality = self.subroutine_table[sub]["avg_quality"]
                        cost = (time ** self.alpha) * (quality ** (2 - self.alpha))
                        sub_cost.append(cost)
                min_index = sub_cost.index(min(sub_cost))
                sr_id = subroutines[min_index]
                return sr_id, self.subroutine_table[sr_id]["tools"]
        return None, None

    def _execute_subroutine(self, tools, subtask, current_state):
        trace = {
            "subtask": subtask, "path": tools, "tools_executed": [], "tool_outputs": {},
            "success": False, "cost": 0, "quality": 1.0, "context_features": {}
        }
        temp_state = current_state.copy()
        metadata = extract_metadata_from_node(subtask)
        temp_state.update(metadata)

        print(f"Starting subroutine for '{subtask}' with initial quality: {trace['quality']:.4f}")
        for tool_name in tools:
            try:
                print(f"Executing tool: {tool_name}")
                tool = self.pipeline.load_tool(tool_name)
                input_spec = self.pipeline._get_tool_input_spec(tool_name)
                tool_inputs = {key: temp_state.get(key) for key in input_spec}
                tool_inputs.update(metadata)
                
                if any(inp not in tool_inputs or tool_inputs[inp] is None for inp in input_spec):
                    print(f"Missing inputs for tool {tool_name}; skipping.")
                    return False, current_state, trace
                
                tool_inputs = {k: v for k, v in tool_inputs.items() if k in input_spec}
                result = tool.process(**tool_inputs)
                exec_time = result.get("execution_time", 0)

                trace["tools_executed"].append(tool_name)
                trace["cost"] += exec_time

                subtask_name_clean = metadata["subtask_name"]
                required_keys = self.trace_requirements.get('subtasks', {}).get(subtask_name_clean, {}).get('trace', [])

                if isinstance(result, dict):
                    filtered_outputs_for_node = {
                        key: value for key, value in result.items()
                        if key in required_keys and key != "image" and not isinstance(value, Image.Image)
                    }
                    if filtered_outputs_for_node:
                        trace["tool_outputs"][tool_name] = filtered_outputs_for_node

                if isinstance(result, dict):
                    if "image" in result and result["image"] is not None:
                        temp_state["image"] = result["image"]
                    for k, v in result.items():
                        if k != "image":
                            temp_state[k] = v
                    output_image = temp_state.get("image")
                else:
                    temp_state["image"] = result
                    output_image = result

                quality = self._check_quality(tool_name, output_image, subtask, temp_state)
                print(f" -> Quality after '{tool_name}': {quality:.4f}. Cumulative subroutine quality: {trace['quality'] * quality:.4f}")
                trace["quality"] *= quality
                if quality < self.quality_threshold:
                    print(f"Quality check failed for {tool_name} with quality {quality:.2f}")
                    return False, current_state, trace
            except Exception as e:
                print(f"Error executing tool {tool_name}: {e}")
                return False, current_state, trace
        
        trace["success"] = True
        print(f"Final subroutine quality for '{subtask}': {trace['quality']:.4f}")
        return True, temp_state, trace

    def _check_quality(self, tool_name, output_image, subtask, state):
        next_tool = subtask
        original_image = state.get("original_image", state.get("image"))
        prompt = state.get("prompt", "")
        tool_result = state.get(subtask, {})
        bounding_box = state.get("bounding_boxes", None)
        pipeline_state = state
        current_path = state.get("current_path", [])
        local_memory = state.get("local_memory", {})
        
        quality = compute_quality_dynamic(
            next_tool, tool_name, original_image, output_image, prompt,
            tool_result, bounding_box, pipeline_state, current_path, local_memory
        )
        print(f"   [Quality Check] Score for '{tool_name}': {quality:.4f}")
        return quality

    def execute_fast_slow_planning(self, image_path, prompt_text, output_chain="Chain.json", output_image="final_output.png"):
        img = Image.open(image_path)
        output_path = output_image
        original_inputs = {"image": img}

        print("Generating subtask chain...")
        subtask_chain = generate_subtask_chain(self.llm_api_key, image_path, prompt_text)
        with open(output_chain, "w") as f:
            json.dump(subtask_chain, f, indent=4)
        print(f"Subtask chain saved to {output_chain}")

        subroutine_chain = generate_subroutine(self.llm_api_key, output_chain, image_path, prompt_text, self.subroutine_table)
        subroutine_chain_path = "Sub.json"
        with open(subroutine_chain_path, "w") as f:
            json.dump(subroutine_chain, f, indent=4)
        print(f"Subroutine chain saved to {subroutine_chain_path}")

        subtask_chain = [node["subtask"] for node in subtask_chain["subtask_chain"]]
        current_state = original_inputs.copy()
        total_cost, total_quality = 0, 1.0
        executed_path = ["Input Image"]

        print(f"\nStarting planning with initial total quality: {total_quality:.4f}")
        for subtask in subtask_chain:
            print(f"\nProcessing subtask: {subtask}")
            context_features = self._extract_context_features(image_path, subtask)
            sr_id, tools = self._select_subroutine(subtask, subroutine_chain, context_features)
            subtask_handled = False
            
            def prepare_trace_features(subtask, context_features):
                metadata = extract_metadata_from_node(subtask)
                subtask_name_clean = metadata.get("subtask_name")
                from_object = metadata.get("from_object")
                target_object = metadata.get("target_object")

                required_keys = self.trace_requirements.get('subtasks', {}).get(subtask_name_clean, {}).get('trace', [])
                
                final_trace_features = {
                    key: value for key, value in context_features.items() if key in required_keys
                }
                
                if from_object and 'object_type' in required_keys:
                    final_trace_features['object_type'] = from_object
                
                if target_object and 'target_object' in required_keys:
                    final_trace_features['target_object'] = target_object
                
                if subtask_name_clean == "Object Recoloration" and from_object and target_object and 'target_color' in required_keys:
                    print("Trace Debugging")
                    print(subtask_name_clean)
                    print(required_keys)
                    print(from_object)
                    print(target_object)
                    color = re.sub(from_object, '', target_object, flags=re.IGNORECASE).strip()
                    print(color)
                    if color:
                        final_trace_features['target_color'] = color
                
                return final_trace_features

            if sr_id and tools:
                print(f"Selected subroutine {sr_id}: {tools}")
                success, new_state, trace = self._execute_subroutine(tools, subtask, current_state)

                trace["context_features"] = prepare_trace_features(subtask, context_features)

                if success:
                    print(f"Fast path for '{subtask}' succeeded with quality {trace['quality']:.4f}.")
                    current_state = new_state
                    total_cost += trace["cost"]
                    total_quality *= trace["quality"]
                    print(f"  -> Updated total quality: {total_quality:.4f}")
                    executed_path.extend([f"{tool} ({subtask})" for tool in tools])
                    append_trace(trace)
                    self.trace_count += 1
                    subtask_handled = True
                else:
                    print(f"Fast plan failed for {subtask}, falling back to slow path")
                    trace["success"] = False
                    append_trace(trace)
                    self.trace_count += 1
            else:
                print(f"No suitable subroutine found for {subtask}, using slow path")

            if not subtask_handled:
                temp_subtask_chain = {"task": f"Sub-plan for {subtask}", "subtask_chain": [{"subtask": subtask, "parent": []}]}
                subtask_subgraph = build_tool_subgraph_from_subtask_chain(temp_subtask_chain)
                print(f"A-Star Subtask Subgraph for {subtask}: {subtask_subgraph}")

                slow_path, slow_state, local_memory, slow_quality = a_star_search(
                    subtask_subgraph, self.alpha, self.quality_threshold,
                    current_state, prompt_text, self.pipeline
                )

                if slow_path and slow_state:
                    print(f"Slow path for '{subtask}' succeeded. Final quality: {slow_quality:.4f}")
                    current_state = slow_state
                    slow_cost = sum(local_memory.get(node, {}).get("execution_time", 0) for node in slow_path[1:])
                    total_cost += slow_cost
                    total_quality *= slow_quality
                    print(f"  -> Updated total quality: {total_quality:.4f}")
                    executed_path.extend(slow_path[1:])
                    
                    subtask_name_clean = extract_metadata_from_node(subtask)["subtask_name"]
                    required_keys = self.trace_requirements.get('subtasks', {}).get(subtask_name_clean, {}).get('trace', [])
                    
                    filtered_tool_outputs = {}
                    for node, outputs in local_memory.items():
                        if node in slow_path[1:]:
                            filtered_outputs_for_node = {k: v for k, v in outputs.items() if k in required_keys}
                            if filtered_outputs_for_node:
                                filtered_tool_outputs[node] = filtered_outputs_for_node

                    filtered_context_features = prepare_trace_features(subtask, context_features)

                    slow_trace = {
                        "subtask": subtask,
                        "path": [node.split(" (")[0] for node in slow_path[1:]],
                        "tools_executed": [node.split(" (")[0] for node in slow_path[1:]],
                        "tool_outputs": filtered_tool_outputs,
                        "success": True, "cost": slow_cost, "quality": slow_quality,
                        "context_features": filtered_context_features
                    }
                    append_trace(slow_trace)
                    self.trace_count += 1
                else:
                    print(f"FATAL: Slow path also failed for {subtask}")
                    return None, executed_path, total_cost, total_quality

        final_image = current_state.get("image")
        if final_image:
            final_image.save(output_path)
            print(f"Final output saved at {output_path}")

        if self.trace_count >= TRACE_TRIGGER_COUNT:
            print(f"Triggering inductive reasoning after {self.trace_count} traces")
            refine_rules_if_needed(self.llm_api_key)

        print(f"\nFinal calculated quality for the entire plan: {total_quality:.4f}")
        return final_image, executed_path, total_cost, total_quality

def execute_fast_slow_planning(image_path, prompt_text, output_chain="Chain.json", output_image="final_output.png", alpha=0, quality_threshold=QUALITY_THRESHOLD):
    llm_api_key = os.getenv("OPENAI_API_KEY")
    if not llm_api_key:
        print("Warning: OPENAI_API_KEY not set. Using empty string.")
        llm_api_key = ""
    
    planner = FastSlowPlanner(llm_api_key, alpha, quality_threshold)
    return planner.execute_fast_slow_planning(image_path, prompt_text, output_chain, output_image)
