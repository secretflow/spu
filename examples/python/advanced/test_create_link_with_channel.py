#!/usr/bin/env python3

# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
One-click startup script for HttpChannel testing

This script automates the process of:
1. Starting HTTP server for HttpChannel
2. Starting node services
3. Running example program with HttpChannel configuration
"""

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional


class Colors:
    """ANSI color codes for terminal output"""

    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


class ColoredPrinter:
    """Utility class for colored terminal output"""

    @staticmethod
    def print_info(message: str):
        print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")

    @staticmethod
    def print_success(message: str):
        print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")

    @staticmethod
    def print_warning(message: str):
        print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")

    @staticmethod
    def print_error(message: str):
        print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")


class HttpChannelTestRunner:
    """Main class to manage the HttpChannel testing environment"""

    def __init__(
        self,
        http_port: int = 11450,
        config_file: str = "examples/python/conf/3pc.json",
        example_program: str = "examples/python/millionaire.py",
        cleanup_on_exit: bool = True,
    ):
        self.http_port = http_port
        self.config_file = config_file
        self.example_program = example_program
        self.cleanup_on_exit = cleanup_on_exit

        self.printer = ColoredPrinter()
        self.http_server_process: Optional[subprocess.Popen] = None
        self.node_processes: List[subprocess.Popen] = []
        self.process_group_pids: List[int] = []  # Track process group PIDs

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Register cleanup function to be called on normal exit
        atexit.register(self.cleanup)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.printer.print_info(f"Received signal {signum}, shutting down...")
        self.cleanup()
        sys.exit(0)

    def _check_file_exists(self, filepath: str) -> bool:
        """Check if a file exists"""
        if not os.path.exists(filepath):
            self.printer.print_error(f"File not found: {filepath}")
            return False
        return True

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return True
            except OSError:
                return False

    def _wait_for_port_available(self, port: int, timeout: int = 10) -> bool:
        """Wait for a port to become available"""
        for i in range(timeout):
            if self._is_port_available(port):
                return True
            self.printer.print_info(
                f"Waiting for port {port} to be available... ({i}/{timeout})"
            )
            time.sleep(1)
        return False

    def _wait_for_http_server(self, timeout: int = 10) -> bool:
        """Wait for HTTP server to be ready"""
        url = f"http://localhost:{self.http_port}/stats"
        for i in range(timeout):
            try:
                urllib.request.urlopen(url, timeout=1)
                self.printer.print_success("HTTP server is ready")
                return True
            except urllib.error.URLError:
                self.printer.print_info(f"Waiting for HTTP server... ({i}/{timeout})")
                time.sleep(1)
        return False

    def cleanup(self):
        """Clean up all running processes"""
        if not self.cleanup_on_exit:
            return

        self.printer.print_info("Cleaning up processes...")

        # First try to kill process groups (more thorough)
        for pgid in self.process_group_pids:
            try:
                os.killpg(pgid, signal.SIGTERM)
                self.printer.print_info(f"Sent SIGTERM to process group {pgid}")
                time.sleep(1)  # Give processes time to terminate gracefully
                os.killpg(pgid, signal.SIGKILL)  # Force kill if still running
            except (ProcessLookupError, PermissionError):
                pass  # Process group might not exist anymore

        # Kill HTTP server
        if self.http_server_process:
            self.printer.print_info(
                f"Stopping HTTP server (PID: {self.http_server_process.pid})"
            )
            self.http_server_process.terminate()
            try:
                self.http_server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.http_server_process.kill()
                try:
                    self.http_server_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass

        # Kill node processes
        for process in self.node_processes:
            if process:
                self.printer.print_info(f"Stopping node process (PID: {process.pid})")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass

        # Additional cleanup: find and kill any remaining related processes
        try:
            # Find any remaining python processes that might be related to our test
            result = subprocess.run(
                ['pgrep', '-f', 'http_server.py|nodectl.py'],
                capture_output=True,
                text=True,
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        time.sleep(0.5)
                        os.kill(int(pid), signal.SIGKILL)
                        self.printer.print_info(f"Cleaned up remaining process {pid}")
                    except (ProcessLookupError, PermissionError, ValueError):
                        pass
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        self.printer.print_success("Cleanup completed")

    def start_http_server(self) -> bool:
        """Start the HTTP server for HttpChannel"""
        self.printer.print_info(
            f"Step 1: Starting HTTP server on port {self.http_port}"
        )

        if not self._check_file_exists("examples/python/advanced/http_server.py"):
            return False

        # Wait for port to be available
        if not self._wait_for_port_available(self.http_port):
            self.printer.print_error(f"Port {self.http_port} is not available")
            return False

        # Start HTTP server
        try:
            self.http_server_process = subprocess.Popen(
                [
                    sys.executable,
                    "examples/python/advanced/http_server.py",
                    "--port",
                    str(self.http_port),
                ],
                preexec_fn=os.setsid,
            )  # Create new session/process group
            self.process_group_pids.append(os.getpgid(self.http_server_process.pid))
            self.printer.print_info(
                f"HTTP server started (PID: {self.http_server_process.pid}, PGID: {os.getpgid(self.http_server_process.pid)})"
            )
        except Exception as e:
            self.printer.print_error(f"Failed to start HTTP server: {e}")
            return False

        # Wait for server to be ready
        if not self._wait_for_http_server():
            self.printer.print_error("HTTP server failed to start")
            return False

        return True

    def start_node_services(self) -> bool:
        """Start the node services"""
        self.printer.print_info("Step 2: Starting node services")

        if not self._check_file_exists("examples/python/utils/nodectl.py"):
            return False

        if not self._check_file_exists(self.config_file):
            return False

        try:
            process = subprocess.Popen(
                [sys.executable, "examples/python/utils/nodectl.py", "up"],
                preexec_fn=os.setsid,
            )  # Create new session/process group
            self.node_processes.append(process)
            pgid = os.getpgid(process.pid)
            self.process_group_pids.append(pgid)
            self.printer.print_info(
                f"Node services started (PID: {process.pid}, PGID: {pgid})"
            )
        except Exception as e:
            self.printer.print_error(f"Failed to start node services: {e}")
            return False

        # Give nodes time to start
        time.sleep(3)
        return True

    def run_example_program(self) -> bool:
        """Run the example program"""
        self.printer.print_info("Step 3: Running example program")
        self.printer.print_info("Set SPU_LINK_METHOD=http to use HttpChannel")

        if not self._check_file_exists(self.example_program):
            return False

        # Set environment variable
        env = os.environ.copy()
        env['SPU_LINK_METHOD'] = 'http'

        try:
            self.printer.print_info(f"Running: python {self.example_program}")
            print("=" * 40)

            result = subprocess.run(
                [sys.executable, self.example_program], env=env, cwd=os.getcwd()
            )

            print("=" * 40)

            if result.returncode == 0:
                self.printer.print_success("Example program completed successfully")
                return True
            else:
                self.printer.print_error(
                    f"Example program failed with return code: {result.returncode}"
                )
                return False

        except Exception as e:
            self.printer.print_error(f"Failed to run example program: {e}")
            return False

    def run(self) -> bool:
        """Run the complete test sequence"""
        self.printer.print_info("Starting HttpChannel testing environment...")
        self.printer.print_info(f"HTTP Port: {self.http_port}")
        self.printer.print_info(f"Config File: {self.config_file}")
        self.printer.print_info(f"Example Program: {self.example_program}")
        print()

        # Step 1: Start HTTP server
        if not self.start_http_server():
            return False

        # Step 2: Start node services
        if not self.start_node_services():
            return False

        # Step 3: Run example program
        if not self.run_example_program():
            return False

        if self.cleanup_on_exit:
            self.printer.print_info("Test completed successfully. Cleaning up...")
            # Cleanup will be called automatically by atexit
            time.sleep(2)  # Give user time to see the success message
        else:
            self.printer.print_warning(
                "Cleanup disabled. Processes will continue running."
            )
            self.printer.print_info("Manually clean up processes using:")
            self.printer.print_info(
                f"  pkill -f 'http_server.py --port {self.http_port}'"
            )
            self.printer.print_info("  pkill -f 'nodectl.py up'")

        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="One-click startup script for HttpChannel testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Use defaults
  %(prog)s --port 8080             # Custom HTTP port
  %(prog)s --example other.py      # Custom example program
  %(prog)s --no-cleanup            # Don't cleanup on exit

Environment variables:
  HTTP_PORT          HTTP server port (default: 11450)
  CONFIG_FILE        Config file for nodes (default: examples/python/conf/3pc.json)
  EXAMPLE_PROGRAM    Example program to run (default: examples/python/millionaire.py)
  CLEANUP_ON_EXIT    Cleanup processes on exit (default: true)
        """,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("HTTP_PORT", 11450)),
        help="HTTP server port (default: 11450)",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("CONFIG_FILE", "examples/python/conf/3pc.json"),
        help="Config file for nodes (default: examples/python/conf/3pc.json)",
    )
    parser.add_argument(
        "--example",
        default=os.environ.get("EXAMPLE_PROGRAM", "examples/python/millionaire.py"),
        help="Example program to run (default: examples/python/millionaire.py)",
    )
    parser.add_argument(
        "--no-cleanup", action="store_true", help="Don't cleanup processes on exit"
    )

    args = parser.parse_args()

    # Create and run test runner
    runner = HttpChannelTestRunner(
        http_port=args.port,
        config_file=args.config,
        example_program=args.example,
        cleanup_on_exit=not args.no_cleanup,
    )

    success = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
