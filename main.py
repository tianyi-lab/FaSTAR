import os
import yaml
import json
import time
import importlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

class ToolPipeline:

    def __init__(self, config_path: str, auto_install: bool = False):

        self.config_path = config_path
        self.auto_install = auto_install
        self.tools_config = self._load_config()
        self.loaded_tools = {}
        
    def _load_config(self) -> Dict[str, Any]:

        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config.get("tools", {})
        except Exception as e:
            print(f"Error loading tool configuration: {e}")
            return {}
            
    def _get_tool_input_spec(self, tool_name: str) -> List[str]:

        if tool_name not in self.tools_config:
            return []
            
        return self.tools_config[tool_name].get('inputs', [])
        
    def _get_tool_output_spec(self, tool_name: str) -> List[str]:

        if tool_name not in self.tools_config:
            return []
            
        return self.tools_config[tool_name].get('outputs', [])
        
    def _install_requirements(self, tool_name: str) -> bool:

        if not self.auto_install:
            return False
            
        if tool_name not in self.tools_config:
            return False
            
        requirements_file = self.tools_config[tool_name].get('requirements')
        if not requirements_file:
            return True  # No requirements needed
            
        requirements_path = requirements_file
        if not os.path.exists(requirements_file):
            print(f"Requirements file not found: {requirements_path}")
            return False
            
        try:
            import subprocess
            result = subprocess.run(
                ['pip', 'install', '-r', str(requirements_path)],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"Installed requirements for {tool_name}: {result.stdout}")
            return True
        except Exception as e:
            print(f"Error installing requirements for {tool_name}: {e}")
            return False
            
    def load_tool(self, tool_name: str) -> Any:

        if tool_name in self.loaded_tools:
            return self.loaded_tools[tool_name]
            
        if tool_name not in self.tools_config:
            raise ValueError(f"Tool not found in configuration: {tool_name}")
            
        if self.auto_install and not self._install_requirements(tool_name):
            print(f"Warning: Failed to install requirements for {tool_name}")

        print(f"Loading tool: {tool_name}")
            
        module_path = self.tools_config[tool_name].get('module')
        class_name = self.tools_config[tool_name].get('class')
        
        if not module_path or not class_name:
            raise ValueError(f"Invalid tool configuration for {tool_name}: missing module or class")
            
        print(f"Loading tool {tool_name} from {module_path}.{class_name}")
        try:
            module = importlib.import_module(module_path)
            tool_class = getattr(module, class_name)
            tool_instance = tool_class(self.tools_config[tool_name])
            self.loaded_tools[tool_name] = tool_instance
            return tool_instance
        except Exception as e:
            print(f"Error loading tool {tool_name}: {e}")
            raise
            
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:

        tool = self.load_tool(tool_name)
        
        input_spec = self._get_tool_input_spec(tool_name)
        missing_inputs = [inp for inp in input_spec if inp not in kwargs]
        if missing_inputs:
            raise ValueError(f"Missing inputs for {tool_name}: {missing_inputs}")
            
        start_time = time.time()
        result = tool.process(**kwargs)
        execution_time = time.time() - start_time
        
        if not isinstance(result, dict):
            result = {"image": result}
            
        result["execution_time"] = execution_time
        
        return result
        
    def get_supported_subtasks(self, tool_name: str) -> List[str]:

        if tool_name not in self.tools_config:
            return []
            
        return self.tools_config[tool_name].get('subtasks', [])
        
    def find_tools_for_subtask(self, subtask_name: str) -> List[str]:

        tools = []
        for tool_name, tool_config in self.tools_config.items():
            if 'subtasks' in tool_config and subtask_name in tool_config['subtasks']:
                tools.append(tool_name)
                
        return tools

def load_image(image_path: str) -> Optional[Image.Image]:

    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def save_image(image: Image.Image, output_path: str) -> bool:

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        image.save(output_path)
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def load_json(file_path: str) -> Optional[Dict[str, Any]]:

    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

def save_json(data: Dict[str, Any], file_path: str) -> bool:

    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON: {e}")
        return False

if __name__ == "__main__":

    pipeline = ToolPipeline("configs/tools.yaml", auto_install=True)
    
    print("Available tools:")
    for tool_name in pipeline.tools_config.keys():
        print(f"  - {tool_name}")
        
    print("\nSupported subtasks per tool:")
    for tool_name in pipeline.tools_config.keys():
        subtasks = pipeline.get_supported_subtasks(tool_name)
        print(f"  {tool_name}: {subtasks}")
        
    subtask = "Object Detection"
    tools = pipeline.find_tools_for_subtask(subtask)
    print(f"\nTools for {subtask}: {tools}")
