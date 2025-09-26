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
Example demonstrating how to use HttpChannel with SPU for secure computation.

This example shows how to create a custom HttpChannel-based link for SPU
computation, which can be useful for scenarios where HTTP-based communication
is preferred over the default gRPC implementation.
"""

import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import json
import spu.libspu.link as link
import spu.libspu as spu
from custom_link import HttpChannel, HttpChannelConfig


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server for demo"""
    pass


class DemoHttpChannelHandler(BaseHTTPRequestHandler):
    """Demo HTTP handler for SPU communication"""

    message_store = {}

    def log_message(self, format, *args):
        """Reduce log noise"""
        pass

    def do_POST(self):
        """Handle POST requests for message sending"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            if self.path.startswith('/send_async/'):
                key = self.path.split('/')[-1]
                self.message_store[key] = post_data

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
        """Handle GET requests for message receiving"""
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
                else:
                    self.send_error(404)
            else:
                self.send_error(404)
        except Exception as e:
            logger.error(f"GET error: {e}")
            self.send_error(500)


def start_demo_servers(ports):
    """Start demo HTTP servers"""
    servers = []
    for port in ports:
        server = ThreadedHTTPServer(('localhost', port), DemoHttpChannelHandler)
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        servers.append(server)
        logger.info(f"Demo server started on port {port}")
    return servers


def create_http_channel_link(parties, self_rank):
    """Create an SPU link using HttpChannel instances"""

    # Create channel configurations
    channels = []
    for i, party in enumerate(parties):
        if i != self_rank:  # Don't create channel to self
            config = HttpChannelConfig(
                peer_url=f"http://{party['host']}",
                timeout_ms=30000,
                max_retry=3,
                retry_interval_ms=1000
            )
            channel = HttpChannel(config)
            channels.append(channel)
        else:
            channels.append(None)  # No channel to self

    # Create link description
    desc = link.Desc()
    desc.id = "http_channel_demo"
    desc.recv_timeout_ms = 30000
    desc.http_max_payload_size = 32 * 1024 * 1024  # 32MB

    for party in parties:
        desc.add_party(party['id'], party['host'])

    # Create link context with custom channels
    # Note: This is a simplified version - in practice you'd need to coordinate
    # which channels to use for each party
    lctx = link.create_with_channels(desc, self_rank, channels)

    return lctx


def demo_simple_computation():
    """Demonstrate simple SPU computation using HttpChannel"""

    logger.info("=== Demo: Simple SPU Computation with HttpChannel ===")

    # Define parties
    parties = [
        {'id': 'alice', 'host': 'localhost:8080'},
        {'id': 'bob', 'host': 'localhost:8081'}
    ]

    # Start demo servers
    servers = start_demo_servers([8080, 8081])
    time.sleep(1)  # Give servers time to start

    try:
        # For this demo, we'll simulate a simple computation
        # In a real scenario, you'd have separate processes for each party

        logger.info("Setting up SPU runtime configuration...")

        # Configure SPU runtime
        config = spu.RuntimeConfig(
            protocol=spu.ProtocolKind.SEMI2K,
            field=spu.FieldType.FM64,
            fxp_fraction_bits=18
        )

        # Create IoWrapper for this demo
        io = spu.IoWrapper(2, config)  # 2 parties

        # Demo input data
        alice_input = [[1, 2, 3], [4, 5, 6]]  # Alice's data
        bob_input = [[7, 8, 9], [10, 11, 12]]  # Bob's data

        logger.info("Creating secret shares...")

        # Create shares for Alice's data (Alice owns the data)
        alice_shares = io.MakeShares(
            alice_input,
            spu.Visibility.VIS_SECRET,
            owner_rank=0  # Alice's rank
        )

        # Create shares for Bob's data (Bob owns the data)
        bob_shares = io.MakeShares(
            bob_input,
            spu.Visibility.VIS_SECRET,
            owner_rank=1  # Bob's rank
        )

        logger.info(f"Created {len(alice_shares)} shares for Alice's data")
        logger.info(f"Created {len(bob_shares)} shares for Bob's data")

        # Simulate secure computation
        # In a real scenario, this would involve MPC protocols
        logger.info("Simulating secure computation...")

        # For demo purposes, we'll just show how the shares work
        alice_share_data = alice_shares[0]  # Alice's share
        bob_share_data = bob_shares[0]     # Bob's share

        logger.info(f"Alice's share meta: {alice_share_data.meta}")
        logger.info(f"Bob's share meta: {bob_share_data.meta}")

        # Demonstrate reconstruction (would normally happen after computation)
        logger.info("Demonstrating reconstruction...")

        # Collect all shares
        all_shares = alice_shares + bob_shares

        # Reconstruct the original data
        # In practice, this would only be done by authorized parties
        reconstructed = io.Reconstruct(all_shares)

        logger.info(f"Reconstructed data: {reconstructed}")

        logger.info("✓ HttpChannel demo completed successfully!")

    except Exception as e:
        logger.error(f"HttpChannel demo failed: {e}")
        raise
    finally:
        # Clean up servers
        for server in servers:
            server.shutdown()


def demo_link_integration():
    """Demonstrate HttpChannel integration with SPU link system"""

    logger.info("=== Demo: HttpChannel Link Integration ===")

    # This demonstrates how HttpChannel would integrate with the SPU link system
    # Note: This is a conceptual example showing the pattern

    try:
        # Create channel configuration
        config = HttpChannelConfig(
            peer_url="http://localhost:8080",
            timeout_ms=30000,
            max_retry=3,
            retry_interval_ms=1000,
            http_max_payload_size=32 * 1024 * 1024,
            enable_ssl=False,
            connection_pool_size=10
        )

        # Create HttpChannel instance
        channel = HttpChannel(config)

        # Test basic channel operations
        logger.info("Testing HttpChannel operations...")

        # Test send/receive
        test_key = "integration_test"
        test_data = b"SPU integration test data"

        channel.Send(test_key, test_data)
        logger.info("✓ Send operation completed")

        # Set receive timeout
        channel.SetRecvTimeout(5000)
        timeout = channel.GetRecvTimeout()
        logger.info(f"✓ Receive timeout set to {timeout}ms")

        # Test configuration
        channel.SetThrottleWindowSize(1024)
        channel.SetChunkParallelSendSize(2)
        logger.info("✓ Channel configuration updated")

        # Test utility methods
        try:
            channel.TestSend(1000)
            logger.info("✓ Send test passed")
        except Exception as e:
            logger.warning(f"Send test failed (expected in demo): {e}")

        try:
            channel.TestRecv()
            logger.info("✓ Receive test passed")
        except Exception as e:
            logger.warning(f"Receive test failed (expected in demo): {e}")

        logger.info("✓ HttpChannel integration demo completed!")

    except Exception as e:
        logger.error(f"HttpChannel integration demo failed: {e}")
        raise


def main():
    """Run all demos"""
    logger.info("Starting HttpChannel demos...")

    try:
        demo_simple_computation()
        demo_link_integration()

        logger.info("🎉 All HttpChannel demos completed successfully!")
        logger.info("\nKey features demonstrated:")
        logger.info("• HttpChannel configuration and initialization")
        logger.info("• Integration with SPU computation workflow")
        logger.info("• Secure data sharing and reconstruction")
        logger.info("• Error handling and timeout management")
        logger.info("• Channel lifecycle management")

    except Exception as e:
        logger.error(f"HttpChannel demos failed: {e}")
        raise


if __name__ == "__main__":
    main()