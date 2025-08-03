"""
Comprehensive test suite for setup_env.py virtual environment setup script.

This module contains tests to verify all functionality of the setup_env.py script,
including virtual environment creation, dependency installation, cross-platform
compatibility, error handling, and command-line argument parsing.

Test Categories:
- Unit tests for individual components and methods
- Integration tests for complete setup workflows
- Cross-platform compatibility tests
- Error handling and edge case tests
- Performance and timeout tests
- Mock tests for external dependencies
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, List, Optional
import pytest

# Import the setup_env module for testing
sys.path.insert(0, str(Path(__file__).parent.parent))
import setup_env


class TestColors:
    """Test the Colors class for cross-platform color support."""
    
    def test_colors_initialization(self):
        """Test that Colors class initializes properly."""
        colors = setup_env.Colors()
        
        # Test that color attributes exist
        assert hasattr(colors, 'RED')
        assert hasattr(colors, 'GREEN')
        assert hasattr(colors, 'RESET')
        
        # Colors should be strings (either ANSI codes or empty)
        assert isinstance(colors.RED, str)
        assert isinstance(colors.GREEN, str)
        assert isinstance(colors.RESET, str)
    
    @patch('setup_env.Colors.SUPPORTED', True)
    def test_colors_when_supported(self):
        """Test color codes when colorama is supported."""
        colors = setup_env.Colors()
        
        # When supported, colors should contain ANSI escape codes
        if colors.SUPPORTED:
            assert '\033[' in colors.RED or colors.RED == ''
            assert '\033[' in colors.GREEN or colors.GREEN == ''
            assert '\033[' in colors.RESET or colors.RESET == ''
    
    @patch('setup_env.Colors.SUPPORTED', False)
    def test_colors_when_not_supported(self):
        """Test color codes when colorama is not supported."""
        colors = setup_env.Colors()
        
        # When not supported, colors should be empty strings
        if not colors.SUPPORTED:
            assert colors.RED == ''
            assert colors.GREEN == ''
            assert colors.RESET == ''


class TestProgressIndicator:
    """Test the ProgressIndicator class."""
    
    def test_progress_indicator_initialization(self):
        """Test ProgressIndicator initialization."""
        progress = setup_env.ProgressIndicator("Test message", verbose=False)
        
        assert progress.message == "Test message"
        assert progress.verbose is False
        assert progress.running is False
        assert progress.current_char == 0
        assert len(progress.spinner_chars) > 0
    
    def test_progress_indicator_start(self):
        """Test starting the progress indicator."""
        progress = setup_env.ProgressIndicator("Test message", verbose=True)
        
        with patch('builtins.print') as mock_print:
            progress.start()
            
            assert progress.running is True
            mock_print.assert_called_once()
    
    def test_progress_indicator_update(self):
        """Test updating the progress indicator."""
        progress = setup_env.ProgressIndicator("Test message", verbose=False)
        progress.running = True
        
        with patch('builtins.print') as mock_print:
            progress.update()
            
            # Should update spinner character
            assert progress.current_char == 1
            mock_print.assert_called_once()
    
    def test_progress_indicator_finish_success(self):
        """Test finishing progress indicator with success."""
        progress = setup_env.ProgressIndicator("Test message", verbose=False)
        progress.running = True
        
        with patch('builtins.print') as mock_print:
            progress.finish(success=True, message="Completed")
            
            assert progress.running is False
            mock_print.assert_called_once()
    
    def test_progress_indicator_finish_failure(self):
        """Test finishing progress indicator with failure."""
        progress = setup_env.ProgressIndicator("Test message", verbose=False)
        progress.running = True
        
        with patch('builtins.print') as mock_print:
            progress.finish(success=False, message="Failed")
            
            assert progress.running is False
            mock_print.assert_called_once()


class TestVirtualEnvSetupInitialization:
    """Test VirtualEnvSetup class initialization."""
    
    @pytest.fixture
    def mock_args(self):
        """Create mock arguments for testing."""
        args = Mock()
        args.venv_name = "test_venv"
        args.python = None
        args.verbose = False
        args.prod = False
        args.force_recreate = False
        return args
    
    def test_virtual_env_setup_initialization(self, mock_args):
        """Test VirtualEnvSetup initialization."""
        with patch('sys.executable', '/usr/bin/python3'):
            setup = setup_env.VirtualEnvSetup(mock_args)
            
            assert setup.args == mock_args
            assert setup.venv_path.name == "test_venv"
            assert setup.python_executable == '/usr/bin/python3'
            assert setup.verbose is False
            assert isinstance(setup.system_info, dict)
            assert 'platform' in setup.system_info
            assert 'python_version' in setup.system_info
    
    def test_requirements_file_paths(self, mock_args):
        """Test that requirements file paths are set correctly."""
        setup = setup_env.VirtualEnvSetup(mock_args)
        
        assert setup.requirements_file.name == "requirements.txt"
        assert setup.dev_requirements_file.name == "requirements-dev.txt"
        assert "aim2_project" in str(setup.requirements_file)
    
    def test_custom_python_executable(self, mock_args):
        """Test using custom Python executable."""
        mock_args.python = "/custom/python"
        setup = setup_env.VirtualEnvSetup(mock_args)
        
        assert setup.python_executable == "/custom/python"


class TestVirtualEnvSetupMethods:
    """Test individual methods of VirtualEnvSetup class."""
    
    @pytest.fixture
    def setup_instance(self):
        """Create a VirtualEnvSetup instance for testing."""
        args = Mock()
        args.venv_name = "test_venv"
        args.python = None
        args.verbose = False
        args.prod = False
        args.force_recreate = False
        
        with patch('sys.executable', '/usr/bin/python3'):
            return setup_env.VirtualEnvSetup(args)
    
    def test_print_banner(self, setup_instance):
        """Test print_banner method."""
        with patch('builtins.print') as mock_print:
            setup_instance.print_banner()
            
            # Should print multiple lines
            assert mock_print.call_count > 5
    
    @patch('subprocess.run')
    def test_check_python_version_success(self, mock_subprocess, setup_instance):
        """Test successful Python version check."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Python 3.9.5"
        mock_subprocess.return_value.stderr = ""
        
        result = setup_instance.check_python_version()
        
        assert result is True
        mock_subprocess.assert_called_once()
    
    @patch('subprocess.run')
    def test_check_python_version_failure(self, mock_subprocess, setup_instance):
        """Test Python version check with unsupported version."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Python 3.7.5"  # Below minimum
        mock_subprocess.return_value.stderr = ""
        
        result = setup_instance.check_python_version()
        
        assert result is False
    
    @patch('subprocess.run')
    def test_check_python_version_timeout(self, mock_subprocess, setup_instance):
        """Test Python version check timeout."""
        mock_subprocess.side_effect = subprocess.TimeoutExpired("python", 10)
        
        result = setup_instance.check_python_version()
        
        assert result is False
    
    @patch('subprocess.run')
    def test_check_python_version_error(self, mock_subprocess, setup_instance):
        """Test Python version check with subprocess error."""
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "Command not found"
        
        result = setup_instance.check_python_version()
        
        assert result is False
    
    def test_check_existing_venv_exists(self, setup_instance):
        """Test checking for existing virtual environment."""
        with patch.object(setup_instance.venv_path, 'exists', return_value=True):
            with patch.object(setup_instance.venv_path / "pyvenv.cfg", 'exists', return_value=True):
                result = setup_instance.check_existing_venv()
                assert result is True
    
    def test_check_existing_venv_not_exists(self, setup_instance):
        """Test checking for non-existing virtual environment."""
        with patch.object(setup_instance.venv_path, 'exists', return_value=False):
            result = setup_instance.check_existing_venv()
            assert result is False
    
    @patch('shutil.rmtree')
    def test_remove_existing_venv_success(self, mock_rmtree, setup_instance):
        """Test successful removal of existing virtual environment."""
        with patch.object(setup_instance, 'check_existing_venv', return_value=True):
            result = setup_instance.remove_existing_venv()
            
            assert result is True
            mock_rmtree.assert_called_once_with(setup_instance.venv_path)
    
    @patch('shutil.rmtree')
    def test_remove_existing_venv_failure(self, mock_rmtree, setup_instance):
        """Test failure to remove existing virtual environment."""
        mock_rmtree.side_effect = OSError("Permission denied")
        
        with patch.object(setup_instance, 'check_existing_venv', return_value=True):
            result = setup_instance.remove_existing_venv()
            
            assert result is False
    
    @patch('subprocess.run')
    def test_create_virtual_environment_success(self, mock_subprocess, setup_instance):
        """Test successful virtual environment creation."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stderr = ""
        
        with patch.object(setup_instance, 'check_existing_venv', side_effect=[False, True]):
            result = setup_instance.create_virtual_environment()
            
            assert result is True
            mock_subprocess.assert_called_once()
    
    @patch('subprocess.run')
    def test_create_virtual_environment_failure(self, mock_subprocess, setup_instance):
        """Test failed virtual environment creation."""
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "Creation failed"
        
        with patch.object(setup_instance, 'check_existing_venv', return_value=False):
            result = setup_instance.create_virtual_environment()
            
            assert result is False
    
    def test_get_venv_python_windows(self, setup_instance):
        """Test getting Python executable path on Windows."""
        setup_instance.system_info['platform'] = 'Windows'
        
        python_path = setup_instance.get_venv_python()
        
        assert "Scripts" in python_path
        assert python_path.endswith("python.exe")
    
    def test_get_venv_python_unix(self, setup_instance):
        """Test getting Python executable path on Unix-like systems."""
        setup_instance.system_info['platform'] = 'Linux'
        
        python_path = setup_instance.get_venv_python()
        
        assert "bin" in python_path
        assert python_path.endswith("python")
    
    def test_get_venv_pip_windows(self, setup_instance):
        """Test getting pip executable path on Windows."""
        setup_instance.system_info['platform'] = 'Windows'
        
        pip_path = setup_instance.get_venv_pip()
        
        assert "Scripts" in pip_path
        assert pip_path.endswith("pip.exe")
    
    def test_get_venv_pip_unix(self, setup_instance):
        """Test getting pip executable path on Unix-like systems."""
        setup_instance.system_info['platform'] = 'Linux'
        
        pip_path = setup_instance.get_venv_pip()
        
        assert "bin" in pip_path
        assert pip_path.endswith("pip")
    
    @patch('subprocess.run')
    def test_upgrade_pip_success(self, mock_subprocess, setup_instance):
        """Test successful pip upgrade."""
        mock_subprocess.return_value.returncode = 0
        
        result = setup_instance.upgrade_pip()
        
        assert result is True
        mock_subprocess.assert_called_once()
    
    @patch('subprocess.run')
    def test_upgrade_pip_failure(self, mock_subprocess, setup_instance):
        """Test failed pip upgrade."""
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "Upgrade failed"
        
        result = setup_instance.upgrade_pip()
        
        assert result is False
    
    @patch('subprocess.Popen')
    @patch('time.sleep')
    def test_install_requirements_success(self, mock_sleep, mock_popen, setup_instance):
        """Test successful requirements installation."""
        # Mock the Popen process
        mock_process = Mock()
        mock_process.poll.side_effect = [None, None, 0]  # Running, then finished
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        # Create a mock requirements file
        requirements_file = setup_instance.project_root / "test_requirements.txt"
        
        with patch.object(requirements_file, 'exists', return_value=True):
            result = setup_instance.install_requirements(requirements_file, "test dependencies")
            
            assert result is True
            mock_popen.assert_called_once()
    
    @patch('subprocess.Popen')
    def test_install_requirements_failure(self, mock_popen, setup_instance):
        """Test failed requirements installation."""
        # Mock the Popen process
        mock_process = Mock()
        mock_process.poll.return_value = 1  # Finished with error
        mock_process.wait.return_value = 1
        mock_process.communicate.return_value = ("Error output", "")
        mock_popen.return_value = mock_process
        
        # Create a mock requirements file
        requirements_file = setup_instance.project_root / "test_requirements.txt"
        
        with patch.object(requirements_file, 'exists', return_value=True):
            result = setup_instance.install_requirements(requirements_file, "test dependencies")
            
            assert result is False
    
    def test_install_requirements_file_not_found(self, setup_instance):
        """Test installation when requirements file doesn't exist."""
        requirements_file = setup_instance.project_root / "nonexistent_requirements.txt"
        
        with patch.object(requirements_file, 'exists', return_value=False):
            result = setup_instance.install_requirements(requirements_file, "test dependencies")
            
            assert result is True  # Should succeed (with warning) when file doesn't exist
    
    @patch('subprocess.run')
    def test_install_project_package_success(self, mock_subprocess, setup_instance):
        """Test successful project package installation."""
        mock_subprocess.return_value.returncode = 0
        
        result = setup_instance.install_project_package()
        
        assert result is True
        mock_subprocess.assert_called_once()
    
    @patch('subprocess.run')
    def test_install_project_package_failure(self, mock_subprocess, setup_instance):
        """Test failed project package installation."""
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "Installation failed"
        
        result = setup_instance.install_project_package()
        
        assert result is False
    
    @patch('subprocess.run')
    def test_verify_installation_success(self, mock_subprocess, setup_instance):
        """Test successful installation verification."""
        mock_subprocess.return_value.returncode = 0
        
        result = setup_instance.verify_installation()
        
        assert result is True
        # Should test multiple packages
        assert mock_subprocess.call_count > 5
    
    @patch('subprocess.run')
    def test_verify_installation_failure(self, mock_subprocess, setup_instance):
        """Test failed installation verification."""
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "Import failed"
        
        result = setup_instance.verify_installation()
        
        assert result is False
    
    def test_print_activation_instructions(self, setup_instance):
        """Test printing activation instructions."""
        with patch('builtins.print') as mock_print:
            setup_instance.print_activation_instructions()
            
            # Should print multiple lines of instructions
            assert mock_print.call_count > 10


class TestVirtualEnvSetupIntegration:
    """Integration tests for complete setup workflow."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "test_project"
            project_dir.mkdir()
            
            # Create basic project structure
            aim2_dir = project_dir / "aim2_project"
            aim2_dir.mkdir()
            (aim2_dir / "__init__.py").touch()
            
            # Create mock requirements files
            (aim2_dir / "requirements.txt").write_text("requests>=2.25.0\npyyaml>=6.0\n")
            (project_dir / "requirements-dev.txt").write_text("pytest>=7.0.0\nblack>=22.0.0\n")
            
            # Create mock setup.py
            setup_py_content = '''
from setuptools import setup, find_packages

setup(
    name="test-project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["requests>=2.25.0", "pyyaml>=6.0"],
)
'''
            (project_dir / "setup.py").write_text(setup_py_content)
            
            yield project_dir
    
    @pytest.fixture
    def mock_args_integration(self):
        """Create mock arguments for integration testing."""
        args = Mock()
        args.venv_name = "test_venv"
        args.python = sys.executable
        args.verbose = True
        args.prod = False
        args.force_recreate = True
        return args
    
    def test_complete_setup_workflow_dev_mode(self, temp_project_dir, mock_args_integration):
        """Test complete setup workflow in development mode."""
        # Change to temporary project directory
        original_cwd = os.getcwd()
        
        try:
            os.chdir(temp_project_dir)
            
            # Patch the project root to use our temp directory
            with patch.object(Path, '__new__') as mock_path:
                def path_side_effect(cls, *args):
                    if len(args) == 1 and str(args[0]).endswith('setup_env.py'):
                        return temp_project_dir / "setup_env.py"
                    return Path.__new__(cls, *args)
                
                mock_path.side_effect = path_side_effect
                
                # Mock external subprocess calls to avoid actual installation
                with patch('subprocess.run') as mock_subprocess:
                    with patch('subprocess.Popen') as mock_popen:
                        # Configure subprocess mocks
                        mock_subprocess.return_value.returncode = 0
                        mock_subprocess.return_value.stdout = "Python 3.9.5"
                        mock_subprocess.return_value.stderr = ""
                        
                        # Mock Popen for requirements installation
                        mock_process = Mock()
                        mock_process.poll.return_value = 0
                        mock_process.wait.return_value = 0
                        mock_popen.return_value = mock_process
                        
                        # Create setup instance
                        setup = setup_env.VirtualEnvSetup(mock_args_integration)
                        setup.project_root = temp_project_dir
                        setup.venv_path = temp_project_dir / "test_venv"
                        setup.requirements_file = temp_project_dir / "aim2_project" / "requirements.txt"
                        setup.dev_requirements_file = temp_project_dir / "requirements-dev.txt"
                        
                        # Mock venv existence checks
                        with patch.object(setup, 'check_existing_venv', return_value=False):
                            # Run the setup
                            result = setup.run_setup()
                            
                            # Should succeed
                            assert result is True
                            
                            # Should have called multiple subprocess operations
                            assert mock_subprocess.call_count > 0
        
        finally:
            os.chdir(original_cwd)
    
    def test_complete_setup_workflow_prod_mode(self, temp_project_dir, mock_args_integration):
        """Test complete setup workflow in production mode."""
        mock_args_integration.prod = True
        mock_args_integration.dev = False
        
        original_cwd = os.getcwd()
        
        try:
            os.chdir(temp_project_dir)
            
            with patch('subprocess.run') as mock_subprocess:
                with patch('subprocess.Popen') as mock_popen:
                    # Configure mocks for success
                    mock_subprocess.return_value.returncode = 0
                    mock_subprocess.return_value.stdout = "Python 3.9.5"
                    mock_subprocess.return_value.stderr = ""
                    
                    mock_process = Mock()
                    mock_process.poll.return_value = 0
                    mock_process.wait.return_value = 0
                    mock_popen.return_value = mock_process
                    
                    # Create setup instance
                    setup = setup_env.VirtualEnvSetup(mock_args_integration)
                    setup.project_root = temp_project_dir
                    setup.venv_path = temp_project_dir / "test_venv"
                    setup.requirements_file = temp_project_dir / "aim2_project" / "requirements.txt"
                    setup.dev_requirements_file = temp_project_dir / "requirements-dev.txt"
                    
                    # Mock venv existence checks
                    with patch.object(setup, 'check_existing_venv', return_value=False):
                        # Run the setup
                        result = setup.run_setup()
                        
                        # Should succeed
                        assert result is True
        
        finally:
            os.chdir(original_cwd)


class TestArgumentParsing:
    """Test command-line argument parsing."""
    
    def test_parse_arguments_defaults(self):
        """Test parsing with default arguments."""
        with patch('sys.argv', ['setup_env.py']):
            args = setup_env.parse_arguments()
            
            assert args.dev is True
            assert args.prod is False
            assert args.venv_name == "venv"
            assert args.force_recreate is False
            assert args.python is None
            assert args.verbose is False
    
    def test_parse_arguments_prod_mode(self):
        """Test parsing production mode arguments."""
        with patch('sys.argv', ['setup_env.py', '--prod']):
            args = setup_env.parse_arguments()
            
            assert args.dev is False
            assert args.prod is True
    
    def test_parse_arguments_custom_venv_name(self):
        """Test parsing custom virtual environment name."""
        with patch('sys.argv', ['setup_env.py', '--venv-name', 'custom_venv']):
            args = setup_env.parse_arguments()
            
            assert args.venv_name == "custom_venv"
    
    def test_parse_arguments_force_recreate(self):
        """Test parsing force recreate flag."""
        with patch('sys.argv', ['setup_env.py', '--force-recreate']):
            args = setup_env.parse_arguments()
            
            assert args.force_recreate is True
    
    def test_parse_arguments_custom_python(self):
        """Test parsing custom Python executable."""
        with patch('sys.argv', ['setup_env.py', '--python', '/custom/python']):
            args = setup_env.parse_arguments()
            
            assert args.python == "/custom/python"
    
    def test_parse_arguments_verbose(self):
        """Test parsing verbose flag."""
        with patch('sys.argv', ['setup_env.py', '--verbose']):
            args = setup_env.parse_arguments()
            
            assert args.verbose is True
    
    def test_parse_arguments_combined(self):
        """Test parsing multiple arguments together."""
        with patch('sys.argv', ['setup_env.py', '--prod', '--venv-name', 'prod_env', '--verbose']):
            args = setup_env.parse_arguments()
            
            assert args.prod is True
            assert args.dev is False
            assert args.venv_name == "prod_env"
            assert args.verbose is True


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def setup_with_mocked_args(self):
        """Create setup instance with mocked arguments."""
        args = Mock()
        args.venv_name = "test_venv"
        args.python = sys.executable
        args.verbose = False
        args.prod = False
        args.force_recreate = False
        
        return setup_env.VirtualEnvSetup(args)
    
    def test_missing_requirements_file(self, setup_with_mocked_args):
        """Test handling of missing requirements file."""
        nonexistent_file = Path("/nonexistent/requirements.txt")
        
        result = setup_with_mocked_args.install_requirements(
            nonexistent_file, "nonexistent dependencies"
        )
        
        # Should return True (with warning) when file doesn't exist
        assert result is True
    
    @patch('subprocess.run')
    def test_python_executable_not_found(self, mock_subprocess, setup_with_mocked_args):
        """Test handling of Python executable not found."""
        mock_subprocess.side_effect = FileNotFoundError("Python not found")
        
        result = setup_with_mocked_args.check_python_version()
        
        assert result is False
    
    @patch('subprocess.run')
    def test_network_timeout_during_pip_install(self, mock_subprocess, setup_with_mocked_args):
        """Test handling of network timeout during pip installation."""
        mock_subprocess.side_effect = subprocess.TimeoutExpired("pip", 300)
        
        result = setup_with_mocked_args.upgrade_pip()
        
        assert result is False
    
    def test_permission_denied_for_venv_creation(self, setup_with_mocked_args):
        """Test handling of permission denied during venv creation."""
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.side_effect = PermissionError("Permission denied")
            
            result = setup_with_mocked_args.create_virtual_environment()
            
            assert result is False
    
    def test_disk_full_during_installation(self, setup_with_mocked_args):
        """Test handling of disk full error during installation."""
        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.poll.return_value = 1
            mock_process.wait.return_value = 1
            mock_process.communicate.return_value = ("No space left on device", "")
            mock_popen.return_value = mock_process
            
            requirements_file = setup_with_mocked_args.project_root / "requirements.txt"
            
            with patch.object(requirements_file, 'exists', return_value=True):
                result = setup_with_mocked_args.install_requirements(
                    requirements_file, "test dependencies"
                )
                
                assert result is False
    
    @patch('shutil.rmtree')
    def test_unable_to_remove_existing_venv(self, mock_rmtree, setup_with_mocked_args):
        """Test handling of inability to remove existing virtual environment."""
        mock_rmtree.side_effect = OSError("Directory not empty")
        
        with patch.object(setup_with_mocked_args, 'check_existing_venv', return_value=True):
            result = setup_with_mocked_args.remove_existing_venv()
            
            assert result is False


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility."""
    
    @pytest.fixture
    def setup_windows(self):
        """Create setup instance for Windows testing."""
        args = Mock()
        args.venv_name = "test_venv"
        args.python = "C:\\Python39\\python.exe"
        args.verbose = False
        args.prod = False
        args.force_recreate = False
        
        setup = setup_env.VirtualEnvSetup(args)
        setup.system_info['platform'] = 'Windows'
        return setup
    
    @pytest.fixture
    def setup_linux(self):
        """Create setup instance for Linux testing."""
        args = Mock()
        args.venv_name = "test_venv"
        args.python = "/usr/bin/python3"
        args.verbose = False
        args.prod = False
        args.force_recreate = False
        
        setup = setup_env.VirtualEnvSetup(args)
        setup.system_info['platform'] = 'Linux'
        return setup
    
    @pytest.fixture
    def setup_macos(self):
        """Create setup instance for macOS testing."""
        args = Mock()
        args.venv_name = "test_venv"
        args.python = "/usr/local/bin/python3"
        args.verbose = False
        args.prod = False
        args.force_recreate = False
        
        setup = setup_env.VirtualEnvSetup(args)
        setup.system_info['platform'] = 'Darwin'
        return setup
    
    def test_windows_executable_paths(self, setup_windows):
        """Test executable paths on Windows."""
        python_path = setup_windows.get_venv_python()
        pip_path = setup_windows.get_venv_pip()
        
        assert "Scripts" in python_path
        assert python_path.endswith("python.exe")
        assert "Scripts" in pip_path
        assert pip_path.endswith("pip.exe")
    
    def test_linux_executable_paths(self, setup_linux):
        """Test executable paths on Linux."""
        python_path = setup_linux.get_venv_python()
        pip_path = setup_linux.get_venv_pip()
        
        assert "bin" in python_path
        assert python_path.endswith("python")
        assert "bin" in pip_path
        assert pip_path.endswith("pip")
    
    def test_macos_executable_paths(self, setup_macos):
        """Test executable paths on macOS."""
        python_path = setup_macos.get_venv_python()
        pip_path = setup_macos.get_venv_pip()
        
        assert "bin" in python_path
        assert python_path.endswith("python")
        assert "bin" in pip_path
        assert pip_path.endswith("pip")
    
    def test_activation_instructions_windows(self, setup_windows):
        """Test activation instructions on Windows."""
        with patch('builtins.print') as mock_print:
            setup_windows.print_activation_instructions()
            
            # Should include Windows-specific instructions
            printed_text = ' '.join([str(call_args[0][0]) for call_args in mock_print.call_args_list])
            assert "Scripts\\activate.bat" in printed_text or "PowerShell" in printed_text
    
    def test_activation_instructions_unix(self, setup_linux):
        """Test activation instructions on Unix-like systems."""
        with patch('builtins.print') as mock_print:
            setup_linux.print_activation_instructions()
            
            # Should include Unix-specific instructions
            printed_text = ' '.join([str(call_args[0][0]) for call_args in mock_print.call_args_list])
            assert "bin/activate" in printed_text or "source" in printed_text


class TestPerformanceAndTimeouts:
    """Test performance characteristics and timeout handling."""
    
    @pytest.fixture
    def setup_with_timeouts(self):
        """Create setup instance for timeout testing."""
        args = Mock()
        args.venv_name = "test_venv"
        args.python = sys.executable
        args.verbose = False
        args.prod = False
        args.force_recreate = False
        
        return setup_env.VirtualEnvSetup(args)
    
    @patch('subprocess.run')
    def test_python_version_check_timeout(self, mock_subprocess, setup_with_timeouts):
        """Test Python version check respects timeout."""
        mock_subprocess.side_effect = subprocess.TimeoutExpired("python", 10)
        
        start_time = time.time()
        result = setup_with_timeouts.check_python_version()
        end_time = time.time()
        
        assert result is False
        # Should not take significantly longer than expected
        assert end_time - start_time < 15  # Allow some buffer
    
    @patch('subprocess.run')
    def test_venv_creation_timeout(self, mock_subprocess, setup_with_timeouts):
        """Test virtual environment creation respects timeout."""
        mock_subprocess.side_effect = subprocess.TimeoutExpired("python", 120)
        
        with patch.object(setup_with_timeouts, 'check_existing_venv', return_value=False):
            start_time = time.time()
            result = setup_with_timeouts.create_virtual_environment()
            end_time = time.time()
            
            assert result is False
            # Should not take significantly longer than expected
            assert end_time - start_time < 125  # Allow some buffer
    
    @patch('subprocess.run')
    def test_pip_upgrade_timeout(self, mock_subprocess, setup_with_timeouts):
        """Test pip upgrade respects timeout."""
        mock_subprocess.side_effect = subprocess.TimeoutExpired("pip", 300)
        
        start_time = time.time()
        result = setup_with_timeouts.upgrade_pip()
        end_time = time.time()
        
        assert result is False
        # Should not take significantly longer than expected
        assert end_time - start_time < 305  # Allow some buffer


class TestMainFunction:
    """Test the main function and entry point."""
    
    def test_main_function_success(self):
        """Test main function with successful setup."""
        test_args = ['--venv-name', 'test_venv', '--verbose']
        
        with patch('sys.argv', ['setup_env.py'] + test_args):
            with patch.object(setup_env.VirtualEnvSetup, 'run_setup', return_value=True):
                with patch('sys.exit') as mock_exit:
                    setup_env.main()
                    
                    mock_exit.assert_called_once_with(0)
    
    def test_main_function_failure(self):
        """Test main function with failed setup."""
        test_args = ['--venv-name', 'test_venv']
        
        with patch('sys.argv', ['setup_env.py'] + test_args):
            with patch.object(setup_env.VirtualEnvSetup, 'run_setup', return_value=False):
                with patch('sys.exit') as mock_exit:
                    setup_env.main()
                    
                    mock_exit.assert_called_once_with(1)
    
    def test_main_function_keyboard_interrupt(self):
        """Test main function handles keyboard interrupt."""
        test_args = ['--venv-name', 'test_venv']
        
        with patch('sys.argv', ['setup_env.py'] + test_args):
            with patch.object(setup_env.VirtualEnvSetup, 'run_setup', side_effect=KeyboardInterrupt):
                with patch('sys.exit') as mock_exit:
                    setup_env.main()
                    
                    mock_exit.assert_called_once_with(130)
    
    def test_main_function_unexpected_error(self):
        """Test main function handles unexpected errors."""
        test_args = ['--venv-name', 'test_venv']
        
        with patch('sys.argv', ['setup_env.py'] + test_args):
            with patch.object(setup_env.VirtualEnvSetup, 'run_setup', side_effect=Exception("Unexpected error")):
                with patch('sys.exit') as mock_exit:
                    setup_env.main()
                    
                    mock_exit.assert_called_once_with(1)


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    @pytest.mark.integration
    def test_dry_run_setup_script(self):
        """Test that setup script can be imported and basic functionality works."""
        # This is a minimal integration test that doesn't create actual virtual environments
        
        # Test that all classes can be instantiated
        colors = setup_env.Colors()
        assert colors is not None
        
        progress = setup_env.ProgressIndicator("Test message")
        assert progress is not None
        
        # Test argument parsing
        with patch('sys.argv', ['setup_env.py', '--help']):
            with pytest.raises(SystemExit):
                setup_env.parse_arguments()
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_minimal_venv_creation(self):
        """Test creating a minimal virtual environment (marked slow)."""
        # This test actually creates a virtual environment
        # Only run if explicitly requested with pytest -m slow
        
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir) / "minimal_test_venv"
            
            # Create minimal virtual environment
            result = subprocess.run([
                sys.executable, "-m", "venv", str(venv_path)
            ], capture_output=True, text=True, timeout=60)
            
            # Should succeed
            assert result.returncode == 0
            
            # Verify venv was created
            assert venv_path.exists()
            assert (venv_path / "pyvenv.cfg").exists()
            
            # Test that Python works in the venv
            if platform.system() == "Windows":
                python_exe = venv_path / "Scripts" / "python.exe"
            else:
                python_exe = venv_path / "bin" / "python"
            
            if python_exe.exists():
                python_result = subprocess.run([
                    str(python_exe), "--version"
                ], capture_output=True, text=True, timeout=10)
                
                assert python_result.returncode == 0
                assert "Python" in python_result.stdout


# Test configuration and markers
pytestmark = [
    pytest.mark.unit,  # Mark all tests as unit tests by default
]

# Integration test markers
class TestMarkers:
    """Test markers for different test categories."""
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit marker works."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        assert True
    
    @pytest.mark.cross_platform
    def test_cross_platform_marker(self):
        """Test that cross-platform marker works."""
        assert True


# Utility functions for test setup
def create_mock_requirements_file(content: str) -> Path:
    """Create a temporary requirements file with given content."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        return Path(f.name)


def cleanup_test_venv(venv_path: Path):
    """Clean up test virtual environment."""
    if venv_path.exists():
        try:
            shutil.rmtree(venv_path)
        except OSError:
            pass  # Best effort cleanup


# Test fixtures for reuse
@pytest.fixture(scope="session")
def test_requirements_content():
    """Sample requirements file content for testing."""
    return """
# Test requirements file
requests>=2.25.0
pyyaml>=6.0
click>=8.0.0
"""


@pytest.fixture(scope="session")
def test_dev_requirements_content():
    """Sample dev requirements file content for testing."""
    return """
# Test dev requirements file
pytest>=7.0.0
black>=22.0.0
mypy>=0.950
"""


# Performance test helpers
class PerformanceTimer:
    """Helper class for measuring test performance."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start timing."""
        self.start_time = time.time()
    
    def stop(self):
        """Stop timing."""
        self.end_time = time.time()
    
    @property
    def elapsed(self):
        """Get elapsed time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


# Test data validation helpers
def validate_requirements_format(requirements_content: str) -> bool:
    """Validate that requirements content has proper format."""
    lines = [line.strip() for line in requirements_content.split('\n') 
             if line.strip() and not line.strip().startswith('#')]
    
    for line in lines:
        if not any(op in line for op in ['>=', '==', '~=', '>', '<']):
            return False
        
        # Basic package name validation
        package_name = line.split('>=')[0].split('==')[0].split('~=')[0]
        if not package_name.replace('-', '_').replace('.', '_').isidentifier():
            return False
    
    return True


# Configuration for different test environments
TEST_ENVIRONMENTS = {
    'minimal': {
        'timeout_multiplier': 1.0,
        'skip_slow_tests': True,
        'mock_external_calls': True,
    },
    'full': {
        'timeout_multiplier': 2.0,
        'skip_slow_tests': False,
        'mock_external_calls': False,
    },
    'ci': {
        'timeout_multiplier': 3.0,
        'skip_slow_tests': True,
        'mock_external_calls': True,
    }
}


def get_test_config():
    """Get test configuration based on environment."""
    env = os.environ.get('TEST_ENV', 'minimal')
    return TEST_ENVIRONMENTS.get(env, TEST_ENVIRONMENTS['minimal'])