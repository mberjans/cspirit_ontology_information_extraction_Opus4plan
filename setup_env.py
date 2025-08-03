#!/usr/bin/env python3
"""
AIM2 Ontology Information Extraction Project - Virtual Environment Setup Script

This script provides a comprehensive, cross-platform setup for the AIM2 project virtual environment.
It handles Python version checking, virtual environment creation, dependency installation,
and provides detailed progress feedback.

Features:
- Cross-platform compatibility (Windows, macOS, Linux)
- Python version validation (requires 3.8+)
- Virtual environment creation and activation
- Dependency installation from requirements files
- Development vs production setup modes
- Comprehensive error handling and recovery
- Progress indicators and status messages
- Idempotent operation (safe to run multiple times)

Usage:
    python setup_env.py [options]

Options:
    --dev              Install development dependencies (default)
    --prod             Install production dependencies only
    --venv-name NAME   Specify virtual environment name (default: venv)
    --force-recreate   Force recreation of existing virtual environment
    --python PATH      Specify Python executable path
    --verbose          Enable verbose output
    --help             Show this help message

Examples:
    python setup_env.py                    # Development setup with default settings
    python setup_env.py --prod             # Production setup only
    python setup_env.py --venv-name myenv  # Use custom environment name
    python setup_env.py --force-recreate   # Recreate existing environment
"""

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple


class Colors:
    """Cross-platform color support for terminal output."""
    
    # Initialize colorama for Windows support
    try:
        import colorama
        colorama.init()
        SUPPORTED = True
    except ImportError:
        SUPPORTED = False
    
    if SUPPORTED:
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        MAGENTA = '\033[95m'
        CYAN = '\033[96m'
        WHITE = '\033[97m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        RESET = '\033[0m'
    else:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ''
        BOLD = UNDERLINE = RESET = ''


class ProgressIndicator:
    """Simple progress indicator for long-running operations."""
    
    def __init__(self, message: str, verbose: bool = False):
        self.message = message
        self.verbose = verbose
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.current_char = 0
        self.running = False
    
    def start(self):
        """Start the progress indicator."""
        if self.verbose:
            print(f"{Colors.CYAN}▶ {self.message}...{Colors.RESET}")
        else:
            print(f"{Colors.CYAN}▶ {self.message}...{Colors.RESET}", end="", flush=True)
        self.running = True
    
    def update(self):
        """Update the spinner (call periodically during operation)."""
        if not self.verbose and self.running:
            print(f"\r{Colors.CYAN}▶ {self.message} {self.spinner_chars[self.current_char]}{Colors.RESET}", 
                  end="", flush=True)
            self.current_char = (self.current_char + 1) % len(self.spinner_chars)
    
    def finish(self, success: bool = True, message: Optional[str] = None):
        """Finish the progress indicator."""
        if self.running:
            status_char = "✓" if success else "✗"
            status_color = Colors.GREEN if success else Colors.RED
            final_message = message or ("Done" if success else "Failed")
            
            if not self.verbose:
                print(f"\r{status_color}{status_char} {self.message} - {final_message}{Colors.RESET}")
            else:
                print(f"{status_color}{status_char} {final_message}{Colors.RESET}")
            
            self.running = False


class VirtualEnvSetup:
    """Main class for virtual environment setup and management."""
    
    def __init__(self, args):
        self.args = args
        self.project_root = Path(__file__).parent.resolve()
        self.venv_path = self.project_root / args.venv_name
        self.python_executable = args.python or sys.executable
        self.verbose = args.verbose
        
        # Requirements file paths
        self.requirements_file = self.project_root / "aim2_project" / "requirements.txt"
        self.dev_requirements_file = self.project_root / "requirements-dev.txt"
        
        self.system_info = {
            'platform': platform.system(),
            'python_version': sys.version_info,
            'architecture': platform.machine(),
        }
    
    def print_banner(self):
        """Print setup banner with project information."""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
        print(f"AIM2 Ontology Information Extraction Project")
        print(f"Virtual Environment Setup Script")
        print(f"{'='*70}{Colors.RESET}")
        print(f"\n{Colors.CYAN}Project Root:{Colors.RESET} {self.project_root}")
        print(f"{Colors.CYAN}Virtual Environment:{Colors.RESET} {self.venv_path}")
        print(f"{Colors.CYAN}Setup Mode:{Colors.RESET} {'Development' if not self.args.prod else 'Production'}")
        print(f"{Colors.CYAN}Python Executable:{Colors.RESET} {self.python_executable}")
        print(f"{Colors.CYAN}Platform:{Colors.RESET} {self.system_info['platform']} ({self.system_info['architecture']})")
        print()
    
    def check_python_version(self) -> bool:
        """Check if Python version meets minimum requirements."""
        progress = ProgressIndicator("Checking Python version", self.verbose)
        progress.start()
        
        try:
            # Get Python version using the specified executable
            result = subprocess.run(
                [self.python_executable, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                progress.finish(False, f"Failed to get Python version: {result.stderr}")
                return False
            
            # Parse version string
            version_str = result.stdout.strip().split()[1]  # "Python 3.x.x" -> "3.x.x"
            version_parts = [int(x) for x in version_str.split('.')]
            current_version = tuple(version_parts[:2])  # (major, minor)
            
            required_version = (3, 8)
            
            if current_version < required_version:
                progress.finish(False, f"Python {version_str} < {'.'.join(map(str, required_version))}")
                print(f"{Colors.RED}✗ Error: Python {'.'.join(map(str, required_version))}+ is required")
                print(f"  Current version: {version_str}")
                print(f"  Please upgrade your Python installation{Colors.RESET}")
                return False
            
            progress.finish(True, f"Python {version_str} ✓")
            return True
            
        except subprocess.TimeoutExpired:
            progress.finish(False, "Python version check timed out")
            return False
        except Exception as e:
            progress.finish(False, f"Error checking Python version: {e}")
            return False
    
    def check_existing_venv(self) -> bool:
        """Check if virtual environment already exists."""
        return self.venv_path.exists() and (self.venv_path / "pyvenv.cfg").exists()
    
    def remove_existing_venv(self) -> bool:
        """Remove existing virtual environment."""
        if not self.check_existing_venv():
            return True
        
        progress = ProgressIndicator("Removing existing virtual environment", self.verbose)
        progress.start()
        
        try:
            import shutil
            shutil.rmtree(self.venv_path)
            progress.finish(True, "Removed successfully")
            return True
        except Exception as e:
            progress.finish(False, f"Failed to remove: {e}")
            return False
    
    def create_virtual_environment(self) -> bool:
        """Create a new virtual environment."""
        if self.check_existing_venv() and not self.args.force_recreate:
            print(f"{Colors.GREEN}✓ Virtual environment already exists: {self.venv_path}{Colors.RESET}")
            return True
        
        if self.args.force_recreate and self.check_existing_venv():
            if not self.remove_existing_venv():
                return False
        
        progress = ProgressIndicator("Creating virtual environment", self.verbose)
        progress.start()
        
        try:
            # Use venv module to create virtual environment
            result = subprocess.run(
                [self.python_executable, "-m", "venv", str(self.venv_path)],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                progress.finish(False, f"Creation failed: {result.stderr}")
                return False
            
            # Verify the virtual environment was created successfully
            if not self.check_existing_venv():
                progress.finish(False, "Virtual environment not found after creation")
                return False
            
            progress.finish(True, "Created successfully")
            return True
            
        except subprocess.TimeoutExpired:
            progress.finish(False, "Virtual environment creation timed out")
            return False
        except Exception as e:
            progress.finish(False, f"Error creating virtual environment: {e}")
            return False
    
    def get_venv_python(self) -> str:
        """Get the path to the Python executable in the virtual environment."""
        if self.system_info['platform'] == 'Windows':
            return str(self.venv_path / "Scripts" / "python.exe")
        else:
            return str(self.venv_path / "bin" / "python")
    
    def get_venv_pip(self) -> str:
        """Get the path to the pip executable in the virtual environment."""
        if self.system_info['platform'] == 'Windows':
            return str(self.venv_path / "Scripts" / "pip.exe")
        else:
            return str(self.venv_path / "bin" / "pip")
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip to the latest version."""
        progress = ProgressIndicator("Upgrading pip", self.verbose)
        progress.start()
        
        try:
            venv_pip = self.get_venv_pip()
            result = subprocess.run(
                [venv_pip, "install", "--upgrade", "pip"],
                capture_output=not self.verbose,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                error_msg = result.stderr if hasattr(result, 'stderr') else "Unknown error"
                progress.finish(False, f"Pip upgrade failed: {error_msg}")
                return False
            
            progress.finish(True, "Pip upgraded successfully")
            return True
            
        except subprocess.TimeoutExpired:
            progress.finish(False, "Pip upgrade timed out")
            return False
        except Exception as e:
            progress.finish(False, f"Error upgrading pip: {e}")
            return False
    
    def install_requirements(self, requirements_file: Path, description: str) -> bool:
        """Install requirements from a specific file."""
        if not requirements_file.exists():
            print(f"{Colors.YELLOW}⚠ Warning: {requirements_file} not found, skipping{Colors.RESET}")
            return True
        
        progress = ProgressIndicator(f"Installing {description}", self.verbose)
        progress.start()
        
        try:
            venv_pip = self.get_venv_pip()
            
            # Install requirements with progress updates
            process = subprocess.Popen(
                [venv_pip, "install", "-r", str(requirements_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Update progress indicator periodically
            while process.poll() is None:
                progress.update()
                time.sleep(0.1)
            
            # Get the final return code
            return_code = process.wait()
            
            if return_code != 0:
                # Get any remaining output
                remaining_output, _ = process.communicate()
                progress.finish(False, f"Installation failed (exit code {return_code})")
                if self.verbose and remaining_output:
                    print(f"{Colors.RED}Error output:\n{remaining_output}{Colors.RESET}")
                return False
            
            progress.finish(True, "Installed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            progress.finish(False, "Installation timed out")
            return False
        except Exception as e:
            progress.finish(False, f"Error installing requirements: {e}")
            return False
    
    def install_project_package(self) -> bool:
        """Install the project package in development mode."""
        progress = ProgressIndicator("Installing project package (editable)", self.verbose)
        progress.start()
        
        try:
            venv_pip = self.get_venv_pip()
            extra_flag = "[dev]" if not self.args.prod else ""
            
            result = subprocess.run(
                [venv_pip, "install", "-e", f".{extra_flag}"],
                cwd=str(self.project_root),
                capture_output=not self.verbose,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                error_msg = result.stderr if hasattr(result, 'stderr') else "Unknown error"
                progress.finish(False, f"Project installation failed: {error_msg}")
                return False
            
            progress.finish(True, "Project package installed")
            return True
            
        except subprocess.TimeoutExpired:
            progress.finish(False, "Project installation timed out")
            return False
        except Exception as e:
            progress.finish(False, f"Error installing project package: {e}")
            return False
    
    def verify_installation(self) -> bool:
        """Verify that key packages are installed correctly."""
        progress = ProgressIndicator("Verifying installation", self.verbose)
        progress.start()
        
        try:
            venv_python = self.get_venv_python()
            
            # Test importing key packages
            test_imports = [
                "transformers",
                "torch", 
                "spacy",
                "pandas",
                "numpy",
                "owlready2",
                "aim2_project"
            ]
            
            for package in test_imports:
                result = subprocess.run(
                    [venv_python, "-c", f"import {package}"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    progress.finish(False, f"Failed to import {package}")
                    if self.verbose:
                        print(f"{Colors.RED}Import error for {package}:\n{result.stderr}{Colors.RESET}")
                    return False
            
            progress.finish(True, "All key packages verified")
            return True
            
        except subprocess.TimeoutExpired:
            progress.finish(False, "Verification timed out")
            return False
        except Exception as e:
            progress.finish(False, f"Error during verification: {e}")
            return False
    
    def print_activation_instructions(self):
        """Print instructions for activating the virtual environment."""
        print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*70}")
        print(f"Setup Complete! 🎉")
        print(f"{'='*70}{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}To activate your virtual environment:{Colors.RESET}")
        
        if self.system_info['platform'] == 'Windows':
            # Windows activation
            print(f"  {Colors.CYAN}Command Prompt:{Colors.RESET}")
            print(f"    {self.venv_path}\\Scripts\\activate.bat")
            print(f"  {Colors.CYAN}PowerShell:{Colors.RESET}")
            print(f"    {self.venv_path}\\Scripts\\Activate.ps1")
        else:
            # Unix-like systems (macOS, Linux)
            print(f"  {Colors.CYAN}Bash/Zsh:{Colors.RESET}")
            print(f"    source {self.venv_path}/bin/activate")
            print(f"  {Colors.CYAN}Fish:{Colors.RESET}")
            print(f"    source {self.venv_path}/bin/activate.fish")
            print(f"  {Colors.CYAN}Csh/Tcsh:{Colors.RESET}")
            print(f"    source {self.venv_path}/bin/activate.csh")
        
        print(f"\n{Colors.BOLD}Available CLI commands:{Colors.RESET}")
        cli_commands = [
            "aim2-ontology-manager",
            "aim2-ner-extractor", 
            "aim2-corpus-builder",
            "aim2-evaluation-benchmarker",
            "aim2-synthetic-generator"
        ]
        
        for cmd in cli_commands:
            print(f"  {Colors.GREEN}•{Colors.RESET} {cmd}")
        
        print(f"\n{Colors.BOLD}To deactivate:{Colors.RESET}")
        print(f"  {Colors.CYAN}deactivate{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}Project Information:{Colors.RESET}")
        print(f"  {Colors.CYAN}Project Root:{Colors.RESET} {self.project_root}")
        print(f"  {Colors.CYAN}Virtual Environment:{Colors.RESET} {self.venv_path}")
        print(f"  {Colors.CYAN}Python Executable:{Colors.RESET} {self.get_venv_python()}")
        print(f"  {Colors.CYAN}Configuration:{Colors.RESET} {self.project_root}/aim2_project/configs/")
        
        if not self.args.prod:
            print(f"\n{Colors.BOLD}Development Tools Available:{Colors.RESET}")
            dev_tools = [
                ("pytest", "Run tests"),
                ("black", "Code formatting"),
                ("mypy", "Type checking"),
                ("jupyter lab", "Start JupyterLab")
            ]
            
            for tool, description in dev_tools:
                print(f"  {Colors.GREEN}•{Colors.RESET} {tool} - {description}")
    
    def run_setup(self) -> bool:
        """Run the complete setup process."""
        self.print_banner()
        
        # Step 1: Check Python version
        if not self.check_python_version():
            return False
        
        # Step 2: Create virtual environment
        if not self.create_virtual_environment():
            return False
        
        # Step 3: Upgrade pip
        if not self.upgrade_pip():
            return False
        
        # Step 4: Install core requirements
        if not self.install_requirements(self.requirements_file, "core dependencies"):
            return False
        
        # Step 5: Install development requirements (if in dev mode)
        if not self.args.prod:
            if not self.install_requirements(self.dev_requirements_file, "development dependencies"):
                return False
        
        # Step 6: Install project package
        if not self.install_project_package():
            return False
        
        # Step 7: Verify installation
        if not self.verify_installation():
            print(f"{Colors.YELLOW}⚠ Warning: Installation verification failed, but setup may still be usable{Colors.RESET}")
        
        # Step 8: Print activation instructions
        self.print_activation_instructions()
        
        return True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AIM2 Project Virtual Environment Setup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_env.py                    # Development setup with default settings
  python setup_env.py --prod             # Production setup only
  python setup_env.py --venv-name myenv  # Use custom environment name
  python setup_env.py --force-recreate   # Recreate existing environment
  python setup_env.py --verbose          # Enable verbose output
        """
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        default=True,
        help="Install development dependencies (default)"
    )
    
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Install production dependencies only"
    )
    
    parser.add_argument(
        "--venv-name",
        default="venv",
        help="Specify virtual environment name (default: venv)"
    )
    
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreation of existing virtual environment"
    )
    
    parser.add_argument(
        "--python",
        help="Specify Python executable path"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle mutually exclusive options
    if args.prod:
        args.dev = False
    
    return args


def main():
    """Main entry point."""
    try:
        args = parse_arguments()
        setup = VirtualEnvSetup(args)
        
        success = setup.run_setup()
        
        if success:
            print(f"\n{Colors.GREEN}✓ Virtual environment setup completed successfully!{Colors.RESET}")
            sys.exit(0)
        else:
            print(f"\n{Colors.RED}✗ Virtual environment setup failed. Please check the error messages above.{Colors.RESET}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Setup interrupted by user.{Colors.RESET}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}✗ Unexpected error during setup: {e}{Colors.RESET}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()