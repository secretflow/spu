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

"""Test script for HttpChannel implementation"""

import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

import json
import spu.libspu.link as link
from custom_link import HttpChannel, HttpChannelConfig


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server for testing"""
    pass


class HttpChannelHandler(BaseHTTPRequestHandler):
    """HTTP request handler for testing HttpChannel"""

    message_store = {}

    def log_message(self, format, *args):
        """Override to reduce noise"""
        pass  # Suppress default logging

    def do_POST(self):
        """Handle POST requests"""
        try:
            if self.path.startswith('/send_async/'):
                key = self.path.split('/')[-1]

                # Parse multipart form data
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)

                # Extract data (simplified - in real implementation use proper multipart parsing)
                # For testing, we'll store the raw data
                self.message_store[key] = post_data

                logger.debug(f"Stored message with key: {key}")

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'ok'}).encode())
            else:
                self.send_error(404)
        except Exception as e:
            logger.error(f"POST error: {e}")
            self.send_error(500)

    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path.startswith('/recv/'):
                key = self.path.split('/')[-1]

                if key in self.message_store:
                    data = self.message_store.pop(key)

                    self.send_response(200)
                    self.send_header('Content-type', 'application/octet-stream')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

                    logger.debug(f"Retrieved and sent message with key: {key}")
                else:
                    self.send_error(404)
            else:
                self.send_error(404)
        except Exception as e:
            logger.error(f"GET error: {e}")
            self.send_error(500)


def start_test_server(port=8080):
    """Start a test HTTP server"""
    server = ThreadedHTTPServer(('localhost', port), HttpChannelHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    logger.info(f"Test server started on port {port}")
    return server


def test_basic_functionality():
    """Test basic send/receive functionality"""
    logger.info("Testing basic HttpChannel functionality...")

    # Start test server
    server = start_test_server(8080)
    time.sleep(0.5)  # Give server time to start

    try:
        # Create HttpChannel configuration
        config = HttpChannelConfig(
            peer_url="http://localhost:8080",
            timeout_ms=5000,
            max_retry=2
        )

        # Create HttpChannel instance
        channel = HttpChannel(config)

        # Test basic send and receive
        test_key = "test_message_1"
        test_data = b"Hello, HttpChannel!"

        # Send data
        logger.info(f"Sending data: {test_data}")
        channel.Send(test_key, test_data)

        # Receive data
        logger.info("Receiving data...")
        received_buffer = channel.Recv(test_key)
        received_data = bytes(received_buffer)

        logger.info(f"Received data: {received_data}")

        # Verify data integrity
        assert received_data == test_data, f"Data mismatch: expected {test_data}, got {received_data}"

        logger.info("✓ Basic functionality test passed")

    except Exception as e:
        logger.error(f"Basic functionality test failed: {e}")
        raise
    finally:
        server.shutdown()


def test_async_functionality():
    """Test async send functionality"""
    logger.info("Testing async HttpChannel functionality...")

    # Start test server
    server = start_test_server(8081)
    time.sleep(0.5)  # Give server time to start

    try:
        # Create HttpChannel configuration
        config = HttpChannelConfig(
            peer_url="http://localhost:8081",
            timeout_ms=5000,
            max_retry=2
        )

        # Create HttpChannel instance
        channel = HttpChannel(config)

        # Test async send
        test_key = "test_async_message"
        test_data = b"Async test message"

        # Send data asynchronously
        logger.info(f"Sending async data: {test_data}")
        channel.SendAsync(test_key, test_data)

        # Give some time for async operation
        time.sleep(0.1)

        # Receive data
        logger.info("Receiving async data...")
        received_buffer = channel.Recv(test_key)
        received_data = bytes(received_buffer)

        logger.info(f"Received async data: {received_data}")

        # Verify data integrity
        assert received_data == test_data, f"Data mismatch: expected {test_data}, got {received_data}"

        logger.info("✓ Async functionality test passed")

    except Exception as e:
        logger.error(f"Async functionality test failed: {e}")
        raise
    finally:
        server.shutdown()


def test_timeout_functionality():
    """Test timeout functionality"""
    logger.info("Testing timeout functionality...")

    # Start test server (won't have the message)
    server = start_test_server(8082)
    time.sleep(0.5)  # Give server time to start

    try:
        # Create HttpChannel configuration with short timeout
        config = HttpChannelConfig(
            peer_url="http://localhost:8082",
            timeout_ms=5000,
            max_retry=1
        )

        # Create HttpChannel instance
        channel = HttpChannel(config)
        channel.SetRecvTimeout(2000)  # 2 second timeout

        # Try to receive non-existent message
        test_key = "non_existent_message"

        logger.info(f"Trying to receive non-existent message: {test_key}")
        try:
            channel.Recv(test_key)
            assert False, "Expected timeout but received data"
        except TimeoutError:
            logger.info("✓ Timeout test passed - correctly timed out")

    except Exception as e:
        logger.error(f"Timeout functionality test failed: {e}")
        raise
    finally:
        server.shutdown()


def test_error_handling():
    """Test error handling"""
    logger.info("Testing error handling...")

    try:
        # Create HttpChannel configuration for non-existent server
        config = HttpChannelConfig(
            peer_url="http://localhost:9999",  # Non-existent port
            timeout_ms=1000,
            max_retry=1
        )

        # Create HttpChannel instance
        channel = HttpChannel(config)

        # Test sending to non-existent server
        test_key = "error_test_message"
        test_data = b"Error test message"

        logger.info("Testing send to non-existent server...")
        try:
            channel.Send(test_key, test_data)
            assert False, "Expected connection error"
        except RuntimeError as e:
            logger.info(f"✓ Correctly caught connection error: {e}")

    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        raise


def test_configuration():
    """Test configuration options"""
    logger.info("Testing configuration...")

    try:
        # Test various configuration options
        config = HttpChannelConfig(
            peer_url="http://localhost:8080",
            timeout_ms=10000,
            max_retry=5,
            retry_interval_ms=500,
            http_max_payload_size=64 * 1024 * 1024,  # 64MB
            enable_ssl=False,
            ssl_verify=True,
            connection_pool_size=20
        )

        channel = HttpChannel(config)

        # Test timeout setting
        channel.SetRecvTimeout(15000)
        assert channel.GetRecvTimeout() == 15000

        # Test throttle settings
        channel.SetThrottleWindowSize(1024)
        channel.SetChunkParallelSendSize(4)

        logger.info("✓ Configuration test passed")

    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        raise


def main():
    """Run all tests"""
    logger.info("Starting HttpChannel tests...")

    try:
        test_basic_functionality()
        test_async_functionality()
        test_timeout_functionality()
        test_error_handling()
        test_configuration()

        logger.info("🎉 All HttpChannel tests passed successfully!")

    except Exception as e:
        logger.error(f"HttpChannel tests failed: {e}")
        raise


if __name__ == "__main__":
    main()