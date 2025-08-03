#!/usr/bin/env python3
"""
FLUX.1 Krea End-to-End Test Suite
Comprehensive testing of all system components
"""

import os
import sys
import time
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime
import json

class FluxE2ETestSuite:
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        self.test_count = 0
        self.passed_count = 0
        self.failed_count = 0
        self.warning_count = 0
        
    def log_test(self, test_name, status, message, details=None):
        """Log test result"""
        self.test_count += 1
        if status == "PASS":
            self.passed_count += 1
            print(f"✅ {test_name}: {message}")
        elif status == "FAIL":
            self.failed_count += 1
            print(f"❌ {test_name}: {message}")
        elif status == "WARN":
            self.warning_count += 1
            print(f"⚠️  {test_name}: {message}")
        
        self.results[test_name] = {
            "status": status,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
    
    def test_syntax_validation(self):
        """Test 1: Syntax validation for all Python files"""
        print("\n🔍 Testing Syntax Validation...")
        
        python_files = [
            "flux_advanced.py", "flux_benchmark.py", "flux_inpaint.py",
            "flux_web_ui.py", "flux_workflow.py", "flux_exceptions.py",
            "flux_model_validator.py", "flux_auth.py", "launch_web_ui.py"
        ]
        
        syntax_results = {}
        for file in python_files:
            if Path(file).exists():
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "py_compile", file],
                        capture_output=True, text=True, timeout=30
                    )
                    if result.returncode == 0:
                        syntax_results[file] = "PASS"
                    else:
                        syntax_results[file] = f"FAIL: {result.stderr[:100]}"
                except Exception as e:
                    syntax_results[file] = f"ERROR: {str(e)}"
            else:
                syntax_results[file] = "MISSING"
        
        failed_files = [f for f, status in syntax_results.items() if not status.startswith("PASS")]
        
        if not failed_files:
            self.log_test("Syntax Validation", "PASS", f"All {len(python_files)} Python files compile successfully", syntax_results)
        else:
            self.log_test("Syntax Validation", "FAIL", f"{len(failed_files)} files have syntax issues", syntax_results)
    
    def test_imports(self):
        """Test 2: Import validation"""
        print("\n📦 Testing Import System...")
        
        import_tests = [
            ("flux_exceptions", "Custom exception classes"),
            ("flux_auth", "Authentication manager"),
            ("flux_model_validator", "Model validator"),
        ]
        
        import_results = {}
        for module, description in import_tests:
            try:
                spec = importlib.util.find_spec(module)
                if spec is not None:
                    imported_module = importlib.import_module(module)
                    import_results[module] = "PASS"
                else:
                    import_results[module] = "FAIL: Module not found"
            except Exception as e:
                import_results[module] = f"FAIL: {str(e)[:100]}"
        
        failed_imports = [m for m, status in import_results.items() if not status.startswith("PASS")]
        
        if not failed_imports:
            self.log_test("Import System", "PASS", "All custom modules import successfully", import_results)
        else:
            self.log_test("Import System", "FAIL", f"{len(failed_imports)} modules failed to import", import_results)
    
    def test_device_detection(self):
        """Test 3: Device detection system"""
        print("\n🖥️  Testing Device Detection...")
        
        try:
            import torch
            
            # Test CUDA detection
            cuda_available = torch.cuda.is_available()
            
            # Test MPS detection
            mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            
            # Determine expected device
            if cuda_available:
                expected_device = "cuda"
            elif mps_available:
                expected_device = "mps"
            else:
                expected_device = "cpu"
            
            device_info = {
                "cuda_available": cuda_available,
                "mps_available": mps_available,
                "expected_device": expected_device,
                "torch_version": torch.__version__
            }
            
            self.log_test("Device Detection", "PASS", f"Device detection working, primary device: {expected_device}", device_info)
            
        except Exception as e:
            self.log_test("Device Detection", "FAIL", f"Device detection failed: {str(e)}")
    
    def test_authentication_system(self):
        """Test 4: Authentication system"""
        print("\n🔑 Testing Authentication System...")
        
        try:
            from flux_auth import auth_manager
            
            # Test authentication status check
            is_authenticated = auth_manager.is_authenticated()
            
            if is_authenticated:
                user_info = auth_manager._user_info
                username = user_info.get('name', 'Unknown') if user_info else 'Unknown'
                
                # Test model access
                access_ok, access_msg = auth_manager.check_model_access()
                
                auth_details = {
                    "authenticated": True,
                    "username": username,
                    "model_access": access_ok,
                    "access_message": access_msg
                }
                
                if access_ok:
                    self.log_test("Authentication System", "PASS", f"Authenticated as {username} with model access", auth_details)
                else:
                    self.log_test("Authentication System", "WARN", f"Authenticated as {username} but no model access: {access_msg}", auth_details)
            else:
                # Try auto-authentication
                success, message = auth_manager.auto_authenticate()
                
                auth_details = {
                    "authenticated": False,
                    "auto_auth_attempted": True,
                    "auto_auth_success": success,
                    "message": message
                }
                
                if success:
                    self.log_test("Authentication System", "PASS", f"Auto-authentication successful: {message}", auth_details)
                else:
                    self.log_test("Authentication System", "WARN", f"No authentication available: {message}", auth_details)
                    
        except Exception as e:
            self.log_test("Authentication System", "FAIL", f"Authentication system error: {str(e)}")
    
    def test_model_validation(self):
        """Test 5: Model validation system"""
        print("\n🎯 Testing Model Validation...")
        
        try:
            from flux_model_validator import ModelValidator
            
            validator = ModelValidator()
            
            # Test local model check
            local_ok, local_msg = validator.check_local_models()
            
            # Test HuggingFace availability
            hf_ok, hf_msg = validator.check_hf_availability()
            
            # Test authentication
            auth_ok, auth_msg = validator.check_hf_authentication()
            
            validation_details = {
                "local_models": {"status": local_ok, "message": local_msg},
                "hf_availability": {"status": hf_ok, "message": hf_msg},
                "hf_authentication": {"status": auth_ok, "message": auth_msg}
            }
            
            if local_ok:
                self.log_test("Model Validation", "PASS", "Local models found and ready", validation_details)
            elif hf_ok and auth_ok:
                self.log_test("Model Validation", "PASS", "Models available via HuggingFace", validation_details)
            else:
                self.log_test("Model Validation", "WARN", "Models not ready - setup required", validation_details)
                
        except Exception as e:
            self.log_test("Model Validation", "FAIL", f"Model validation error: {str(e)}")
    
    def test_advanced_generation_startup(self):
        """Test 6: Advanced generation system startup"""
        print("\n🎨 Testing Advanced Generation System...")
        
        try:
            # Test script exists and has correct permissions
            script_path = Path("flux_advanced.py")
            if not script_path.exists():
                self.log_test("Advanced Generation", "FAIL", "flux_advanced.py not found")
                return
            
            # Test help command
            result = subprocess.run(
                [sys.executable, "flux_advanced.py", "--help"],
                capture_output=True, text=True, timeout=30,
                cwd=os.getcwd(),
                env={**os.environ, "PYTHONPATH": os.getcwd()}
            )
            
            if result.returncode == 0:
                help_output = result.stdout
                required_args = ["--prompt", "--optimization", "--width", "--height"]
                missing_args = [arg for arg in required_args if arg not in help_output]
                
                if not missing_args:
                    self.log_test("Advanced Generation", "PASS", "Script loads and shows proper help", {"help_length": len(help_output)})
                else:
                    self.log_test("Advanced Generation", "WARN", f"Missing arguments in help: {missing_args}")
            else:
                self.log_test("Advanced Generation", "FAIL", f"Help command failed: {result.stderr[:200]}")
                
        except Exception as e:
            self.log_test("Advanced Generation", "FAIL", f"Advanced generation test error: {str(e)}")
    
    def test_web_ui_startup(self):
        """Test 7: Web UI startup"""
        print("\n🌐 Testing Web UI Startup...")
        
        try:
            # Test Gradio import
            import gradio as gr
            gradio_version = gr.__version__
            
            # Test web UI script syntax
            result = subprocess.run(
                [sys.executable, "-c", "import flux_web_ui; print('Web UI imports successfully')"],
                capture_output=True, text=True, timeout=30,
                cwd=os.getcwd(),
                env={**os.environ, "PYTHONPATH": os.getcwd()}
            )
            
            if result.returncode == 0:
                self.log_test("Web UI Startup", "PASS", f"Web UI imports successfully (Gradio {gradio_version})", {"gradio_version": gradio_version})
            else:
                self.log_test("Web UI Startup", "FAIL", f"Web UI import failed: {result.stderr[:200]}")
                
        except ImportError as e:
            self.log_test("Web UI Startup", "FAIL", f"Gradio not available: {str(e)}")
        except Exception as e:
            self.log_test("Web UI Startup", "FAIL", f"Web UI test error: {str(e)}")
    
    def test_benchmark_system(self):
        """Test 8: Benchmark system"""
        print("\n⚡ Testing Benchmark System...")
        
        try:
            # Check if benchmark script exists
            script_path = Path("flux_benchmark.py")
            if not script_path.exists():
                self.log_test("Benchmark System", "FAIL", "flux_benchmark.py not found")
                return
            
            # Test script syntax by importing
            result = subprocess.run(
                [sys.executable, "-c", "import flux_benchmark; print('Benchmark imports successfully')"],
                capture_output=True, text=True, timeout=30,
                cwd=os.getcwd(),
                env={**os.environ, "PYTHONPATH": os.getcwd()}
            )
            
            if result.returncode == 0:
                self.log_test("Benchmark System", "PASS", "Benchmark system imports successfully")
            else:
                self.log_test("Benchmark System", "FAIL", f"Benchmark import failed: {result.stderr[:200]}")
                
        except Exception as e:
            self.log_test("Benchmark System", "FAIL", f"Benchmark test error: {str(e)}")
    
    def test_workflow_system(self):
        """Test 9: Workflow system"""
        print("\n🔄 Testing Workflow System...")
        
        try:
            script_path = Path("flux_workflow.py")
            if not script_path.exists():
                self.log_test("Workflow System", "FAIL", "flux_workflow.py not found")
                return
            
            # Test help command
            result = subprocess.run(
                [sys.executable, "flux_workflow.py", "--help"],
                capture_output=True, text=True, timeout=30,
                cwd=os.getcwd(),
                env={**os.environ, "PYTHONPATH": os.getcwd()}
            )
            
            if result.returncode == 0:
                self.log_test("Workflow System", "PASS", "Workflow system help command works")
            else:
                self.log_test("Workflow System", "FAIL", f"Workflow help failed: {result.stderr[:200]}")
                
        except Exception as e:
            self.log_test("Workflow System", "FAIL", f"Workflow test error: {str(e)}")
    
    def test_desktop_shortcut(self):
        """Test 10: Desktop shortcut"""
        print("\n🖥️  Testing Desktop Shortcut...")
        
        try:
            shortcut_path = Path.home() / "Desktop" / "FLUX Krea Studio.app"
            launcher_path = Path("launch_flux_web.sh")
            
            shortcut_exists = shortcut_path.exists()
            launcher_exists = launcher_path.exists()
            
            shortcut_details = {
                "shortcut_path": str(shortcut_path),
                "shortcut_exists": shortcut_exists,
                "launcher_exists": launcher_exists
            }
            
            if shortcut_exists and launcher_exists:
                self.log_test("Desktop Shortcut", "PASS", "Desktop shortcut and launcher found", shortcut_details)
            elif launcher_exists:
                self.log_test("Desktop Shortcut", "WARN", "Launcher exists but desktop shortcut missing", shortcut_details)
            else:
                self.log_test("Desktop Shortcut", "FAIL", "Neither shortcut nor launcher found", shortcut_details)
                
        except Exception as e:
            self.log_test("Desktop Shortcut", "FAIL", f"Desktop shortcut test error: {str(e)}")
    
    def test_dependencies(self):
        """Test 11: Dependency availability"""
        print("\n📦 Testing Dependencies...")
        
        required_packages = [
            ("torch", "PyTorch"),
            ("PIL", "Pillow"),
            ("numpy", "NumPy"),
            ("gradio", "Gradio"),
            ("huggingface_hub", "HuggingFace Hub"),
            ("psutil", "System utilities")
        ]
        
        dependency_results = {}
        for package, description in required_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'Unknown')
                dependency_results[package] = {"status": "PASS", "version": version}
            except ImportError:
                dependency_results[package] = {"status": "MISSING", "version": None}
            except Exception as e:
                dependency_results[package] = {"status": "ERROR", "error": str(e)}
        
        missing_deps = [pkg for pkg, info in dependency_results.items() if info["status"] != "PASS"]
        
        if not missing_deps:
            self.log_test("Dependencies", "PASS", f"All {len(required_packages)} required packages available", dependency_results)
        else:
            self.log_test("Dependencies", "FAIL", f"{len(missing_deps)} packages missing: {missing_deps}", dependency_results)
    
    def generate_report(self):
        """Generate comprehensive test report"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        report = {
            "test_summary": {
                "total_tests": self.test_count,
                "passed": self.passed_count,
                "failed": self.failed_count,
                "warnings": self.warning_count,
                "duration_seconds": duration.total_seconds(),
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat()
            },
            "test_results": self.results,
            "system_info": {
                "platform": sys.platform,
                "python_version": sys.version,
                "working_directory": os.getcwd()
            }
        }
        
        # Save detailed JSON report
        report_path = Path("e2e_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        self.generate_markdown_report(report)
        
        return report
    
    def generate_markdown_report(self, report_data):
        """Generate human-readable markdown report"""
        
        summary = report_data["test_summary"]
        results = report_data["test_results"]
        
        # Calculate success rate
        success_rate = (summary["passed"] / summary["total_tests"] * 100) if summary["total_tests"] > 0 else 0
        
        markdown_content = f"""# 🧪 FLUX-Krea End-to-End Test Report

## 📊 Test Summary

- **Total Tests:** {summary["total_tests"]}
- **✅ Passed:** {summary["passed"]}
- **❌ Failed:** {summary["failed"]}
- **⚠️  Warnings:** {summary["warnings"]}
- **Success Rate:** {success_rate:.1f}%
- **Duration:** {summary["duration_seconds"]:.1f} seconds
- **Timestamp:** {summary["end_time"]}

## 🎯 Overall Status

"""
        
        if summary["failed"] == 0:
            markdown_content += "🎉 **ALL TESTS PASSED** - System is production ready!\n"
        elif summary["failed"] <= 2:
            markdown_content += "⚠️  **MOSTLY FUNCTIONAL** - Minor issues to address\n"
        else:
            markdown_content += "❌ **NEEDS ATTENTION** - Multiple critical issues found\n"
        
        markdown_content += "\n## 📋 Detailed Results\n\n"
        
        # Sort results by status (FAIL, WARN, PASS)
        status_order = {"FAIL": 0, "WARN": 1, "PASS": 2}
        sorted_results = sorted(results.items(), key=lambda x: status_order.get(x[1]["status"], 3))
        
        for test_name, result in sorted_results:
            status = result["status"]
            message = result["message"]
            
            if status == "PASS":
                emoji = "✅"
            elif status == "WARN":
                emoji = "⚠️ "
            else:
                emoji = "❌"
            
            markdown_content += f"\n### {emoji} {test_name}\n"
            markdown_content += f"**Status:** {status}  \n"
            markdown_content += f"**Result:** {message}  \n"
            
            if result.get("details"):
                markdown_content += f"**Details:** {json.dumps(result['details'], indent=2)}  \n"
            
            markdown_content += f"**Timestamp:** {result['timestamp']}  \n"
        
        # Add recommendations
        markdown_content += "\n## 💡 Recommendations\n\n"
        
        if summary["failed"] == 0 and summary["warnings"] == 0:
            markdown_content += "🎉 **System is fully operational!** All tests passed successfully.\n"
        else:
            if summary["failed"] > 0:
                markdown_content += "🔧 **Critical Issues to Address:**\n"
                failed_tests = [name for name, result in results.items() if result["status"] == "FAIL"]
                for test in failed_tests:
                    markdown_content += f"- Fix {test}: {results[test]['message']}\n"
                markdown_content += "\n"
            
            if summary["warnings"] > 0:
                markdown_content += "⚠️  **Warnings to Consider:**\n"
                warning_tests = [name for name, result in results.items() if result["status"] == "WARN"]
                for test in warning_tests:
                    markdown_content += f"- {test}: {results[test]['message']}\n"
                markdown_content += "\n"
        
        # Save markdown report
        report_path = Path("E2E_TEST_REPORT.md")
        with open(report_path, 'w') as f:
            f.write(markdown_content)
        
        print(f"\n📄 Reports generated:")
        print(f"   - Detailed: e2e_test_report.json")
        print(f"   - Summary: E2E_TEST_REPORT.md")
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting FLUX-Krea End-to-End Test Suite")
        print("=" * 60)
        
        # Run all tests
        self.test_syntax_validation()
        self.test_imports()
        self.test_dependencies()
        self.test_device_detection()
        self.test_authentication_system()
        self.test_model_validation()
        self.test_advanced_generation_startup()
        self.test_web_ui_startup()
        self.test_benchmark_system()
        self.test_workflow_system()
        self.test_desktop_shortcut()
        
        # Generate reports
        print("\n📊 Generating Test Reports...")
        report = self.generate_report()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎯 TEST SUITE COMPLETE")
        print("=" * 60)
        print(f"✅ Passed: {self.passed_count}")
        print(f"⚠️  Warnings: {self.warning_count}")
        print(f"❌ Failed: {self.failed_count}")
        print(f"📊 Success Rate: {(self.passed_count/self.test_count)*100:.1f}%")
        
        return report

if __name__ == "__main__":
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Activate virtual environment if available
    venv_python = script_dir / "flux_env" / "bin" / "python"
    if venv_python.exists():
        print("🐍 Using virtual environment")
        # Note: We're already running in the activated environment
    
    # Run test suite
    test_suite = FluxE2ETestSuite()
    test_suite.run_all_tests()