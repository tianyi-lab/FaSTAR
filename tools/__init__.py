import sys
import subprocess
import pkg_resources
from pathlib import Path
from abc import ABC, abstractmethod
from PIL import Image

class DependencyManager:
    @staticmethod
    def install_requirements(requirements_path: str):
        """Install requirements only if needed"""
        try:
            with open(requirements_path) as f:
                requirements = [str(req) for req in pkg_resources.parse_requirements(f)]
        except FileNotFoundError:
            return

        to_install = []
        for req in requirements:
            try:
                pkg_resources.require(req)
            except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
                to_install.append(req)

        if to_install:
            print(f"Installing missing dependencies: {', '.join(to_install)}")
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "--user", "-r", requirements_path
            ], check=True)

class BaseTool(ABC):
    def __init__(self, config: dict):
        self._install_dependencies(config.get('requirements'))

    def _install_dependencies(self, req_file: str):
        if req_file and Path(req_file).exists():
            DependencyManager.install_requirements(req_file)

    @abstractmethod
    def load_model(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def process(self, input_image: Image, **kwargs) -> Image:
        pass


_all_ = ["BaseTool", "DependencyManager"]