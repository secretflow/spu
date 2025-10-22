#!/usr/bin/env python3
"""
Test script for HttpChannel with multi-node simulation
"""

import spu.libspu.link as link
import time
import sys
import argparse
from http_channel import HttpChannel
from typing import List, Optional
import requests


def create_channels_for_node(
    node_rank: int, total_nodes: int, server_port: int
) -> List[Optional[HttpChannel]]:
    """Create HttpChannels for a node to communicate with all other nodes"""
    channels: List[Optional[HttpChannel]] = []

    for remote_rank in range(total_nodes):
        if remote_rank == node_rank:
            channels.append(None)  # Self position gets None
        else:
            channel = HttpChannel(
                local_rank=node_rank,
                remote_rank=remote_rank,
                base_port=server_port,
                channel_id=f"channel_{node_rank}_{remote_rank}",
            )
            channels.append(channel)
            print(
                f"Node {node_rank}: Created channel to Node {remote_rank} via localhost:{server_port}"
            )

    return channels


def test_basic_http_channel(server_port: int):
    """Test basic HttpChannel functionality"""
    print("=== Test Basic HttpChannel Functionality ===")

    try:
        # Create two channels
        alice = HttpChannel(0, 1, server_port)
        bob = HttpChannel(1, 0, server_port)

        # Test basic communication
        print("Alice sending message to Bob...")
        alice.Send("test_message", b"hello bob via HTTP")

        time.sleep(0.1)  # Give server time to process

        print("Bob receiving message...")
        received = bob.Recv("test_message")
        print(f"Bob received: {received}")

        # Test reverse communication
        print("Bob sending response to Alice...")
        bob.Send("response", b"hello alice via HTTP")

        time.sleep(0.1)

        print("Alice receiving response...")
        response = alice.Recv("response")
        print(f"Alice received: {response}")

        print("✅ Basic HttpChannel test completed")

    except Exception as e:
        print(f"❌ Basic HttpChannel test failed: {e}")


def test_with_create_with_channels(server_port: int):
    """Test HttpChannel with create_with_channels interface"""
    print("\n=== Test HttpChannel with create_with_channels ===")

    try:
        # Create channels for two nodes
        alice_channels = create_channels_for_node(0, 2, server_port)
        bob_channels = create_channels_for_node(1, 2, server_port)

        # Create device description
        desc = link.Desc()
        desc.add_party("party_0", f"127.0.0.1:{server_port}")
        desc.add_party("party_1", f"127.0.0.1:{server_port}")

        # Create device contexts with custom channels
        # For node 0: [None, channel_to_node_1]
        ctx0 = link.create_with_channels(desc, 0, alice_channels)

        # For node 1: [channel_to_node_0, None]
        ctx1 = link.create_with_channels(desc, 1, bob_channels)

        print("✅ create_with_channels interface call successful")
        print(f"Node 0 context - rank: {ctx0.rank}, world_size: {ctx0.world_size}")
        print(f"Node 1 context - rank: {ctx1.rank}, world_size: {ctx1.world_size}")

        # Test context communication
        print("Testing context Send and Recv...")
        test_msg = "hello from node 0 via context"
        ctx0.send(1, test_msg)

        time.sleep(0.1)

        received = ctx1.recv(0)
        print(f"Node 1 received via context: {received}")

        # Test reverse communication
        test_msg_back = "hello from node 1 via context"
        ctx1.send(0, test_msg_back)

        time.sleep(0.1)

        received_back = ctx0.recv(1)
        print(f"Node 0 received via context: {received_back}")

        # Test async send
        ctx0.send_async(1, "async message from node 0")

        time.sleep(0.1)

        received_async = ctx1.recv(0)
        print(f"Node 1 received async: {received_async}")

        print("✅ Context communication test completed")

    except Exception as e:
        print(f"❌ Context communication test failed: {e}")
        import traceback

        traceback.print_exc()


def test_three_node_simulation(server_port: int):
    """Test three-node communication scenario"""
    print("\n=== Test Three-Node Communication ===")

    try:
        total_nodes = 3

        # Create channels for all nodes
        all_channels = {}
        for rank in range(total_nodes):
            all_channels[rank] = create_channels_for_node(
                rank, total_nodes, server_port
            )

        # Create device description for 3 nodes
        desc = link.Desc()
        for i in range(total_nodes):
            desc.add_party(f"party_{i}", f"127.0.0.1:{server_port}")

        # Create contexts for each node
        contexts = []
        for rank in range(total_nodes):
            # create_channels_for_node already returns the correct list with None at self position
            ctx = link.create_with_channels(desc, rank, all_channels[rank])
            contexts.append(ctx)
            print(f"Created context for node {rank}")

        print("✅ All three nodes connected successfully")

        # Test all-to-all communication
        print("Testing all-to-all communication...")

        for sender_rank in range(total_nodes):
            for receiver_rank in range(total_nodes):
                if sender_rank != receiver_rank:
                    message = f"Hello from node {sender_rank} to node {receiver_rank}"
                    contexts[sender_rank].send(receiver_rank, message)
                    print(f"Node {sender_rank} -> Node {receiver_rank}: {message}")

        time.sleep(0.5)  # Give server time to process all messages

        # Receive messages
        for receiver_rank in range(total_nodes):
            for sender_rank in range(total_nodes):
                if sender_rank != receiver_rank:
                    received = contexts[receiver_rank].recv(sender_rank)
                    print(
                        f"Node {receiver_rank} received from Node {sender_rank}: {received}"
                    )

        print("✅ Three-node communication test completed")

    except Exception as e:
        print(f"❌ Three-node communication test failed: {e}")
        import traceback

        traceback.print_exc()


def check_server_availability(server_port: int) -> bool:
    """Check if the HTTP server is available"""
    try:
        response = requests.get(f"http://localhost:{server_port}/stats", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test HttpChannel implementation')
    parser.add_argument(
        '--server-port',
        type=int,
        default=11450,
        help='HTTP server port (default: 11450)',
    )
    parser.add_argument(
        '--test',
        type=str,
        choices=['basic', 'context', 'three-node', 'all'],
        default='all',
        help='Which test to run (default: all)',
    )

    args = parser.parse_args()

    print("Starting HttpChannel Integration Tests...")
    print(f"Server: localhost:{args.server_port}")
    print("=" * 60)

    # Check if server is running
    if not check_server_availability(args.server_port):
        print(f"❌ HTTP server is not running at localhost:{args.server_port}")
        print("Please start the server with:")
        print(f"  python http_server.py --port {args.server_port}")
        sys.exit(1)

    print(f"✅ HTTP server is running at localhost:{args.server_port}")
    print()

    try:
        # Run selected tests
        if args.test in ['basic', 'all']:
            test_basic_http_channel(args.server_port)
            time.sleep(1)

        if args.test in ['context', 'all']:
            test_with_create_with_channels(args.server_port)
            time.sleep(1)

        if args.test in ['three-node', 'all']:
            test_three_node_simulation(args.server_port)

        print("\n" + "=" * 60)
        print("✅ All HttpChannel tests completed!")

    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
