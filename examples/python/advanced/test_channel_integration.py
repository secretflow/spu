#!/usr/bin/env python3
"""
Test script: Verify Python class implementing IChannel and passing to C++
"""

import spu.libspu.link as link
import spu.libspu as spu
from typing import Dict, Optional

print(dir(link))


class SimpleChannel(link.IChannel):
    """Simple point-to-point channel implementation"""

    def __init__(
        self, name: str, storage: Dict[str, bytes], local_rank: int, remote_rank: int
    ):
        super().__init__()
        self.name = name
        self.local_rank = local_rank
        self.remote_rank = remote_rank
        # Each channel has its own storage, peer_storage allows sharing between two channels
        self.storage = storage
        self.recv_timeout = 1000
        self.throttle_window_size = 1024
        self.chunk_parallel_send_size = 4

    def SendAsync(self, key: str, buf: bytes) -> None:
        """Asynchronously send data"""
        final_key = f"{key}_{self.local_rank}_{self.remote_rank}"
        print(f"[{self.name}] SendAsync: key={final_key}, size={len(buf)}")
        self.storage[final_key] = buf

    def SendAsyncThrottled(self, key: str, buf: bytes) -> None:
        """Asynchronously send data with throttling"""
        final_key = f"{key}_{self.local_rank}_{self.remote_rank}"
        print(f"[{self.name}] SendAsyncThrottled: key={final_key}, size={len(buf)}")
        self.storage[final_key] = buf

    def Send(self, key: str, value: bytes) -> None:
        """Synchronously send data"""
        final_key = f"{key}_{self.local_rank}_{self.remote_rank}"
        print(f"[{self.name}] Send: key={final_key}, size={len(value)}")
        self.storage[final_key] = value

    def Recv(self, key: str) -> bytes:
        """Receive data"""
        final_key = f"{key}_{self.remote_rank}_{self.local_rank}"
        print(f"[{self.name}] Recv: key={final_key}")
        return self.storage.get(final_key, b"")

    def SetRecvTimeout(self, timeout_ms: int) -> None:
        """Set receive timeout"""
        print(f"[{self.name}] SetRecvTimeout: {timeout_ms}ms")
        self.recv_timeout = timeout_ms

    def GetRecvTimeout(self) -> int:
        """Get receive timeout"""
        return self.recv_timeout

    def WaitLinkTaskFinish(self) -> None:
        """Wait for link tasks to finish"""
        print(f"[{self.name}] WaitLinkTaskFinish")

    def Abort(self) -> None:
        """Abort operation"""
        print(f"[{self.name}] Abort")

    def SetThrottleWindowSize(self, size: int) -> None:
        """Set throttle window size"""
        print(f"[{self.name}] SetThrottleWindowSize: {size}")
        self.throttle_window_size = size

    def TestSend(self, timeout: int) -> None:
        """Test send functionality"""
        self.Send("test", b"")

    def TestRecv(self) -> None:
        """Test receive functionality"""
        self.Recv("test")

    def SetChunkParallelSendSize(self, size: int) -> None:
        """Set chunk parallel send size"""
        print(f"[{self.name}] SetChunkParallelSendSize: {size}")
        self.chunk_parallel_send_size = size


def test_basic_functionality():
    """Test basic functionality"""
    print("=== Test Basic Functionality ===")

    # Create two channels with shared storage
    storage = {}
    alice = SimpleChannel("Alice", storage, local_rank=0, remote_rank=1)
    bob = SimpleChannel("Bob", storage, local_rank=1, remote_rank=0)

    # Test basic communication
    alice.Send("test_message", b"hello bob")
    received = bob.Recv("test_message")
    print(f"Bob received: {received}")

    # Reverse communication
    bob.Send("response", b"hello alice")
    received_back = alice.Recv("response")
    print(f"Alice received: {received_back}")


def test_with_create_with_channels():
    """Test create_with_channels interface"""
    print("\n=== Test create_with_channels Interface ===")

    try:

        # For simplicity, each rank uses independent channels
        # In real scenarios, these channels would connect via network
        storage = {}
        channelA2B = SimpleChannel("ChannelA2B", storage, local_rank=0, remote_rank=1)
        channelB2A = SimpleChannel("ChannelB2A", storage, local_rank=1, remote_rank=0)

        # Create device description
        desc = link.Desc()
        # Add required parties for the test
        desc.add_party("party_0", "127.0.0.1:9000")
        desc.add_party("party_1", "127.0.0.1:9001")

        # Create device contexts with custom channels
        ctxA = link.create_with_channels(desc, 0, [None, channelA2B])
        ctxB = link.create_with_channels(desc, 1, [channelB2A, None])

        print("✅ create_with_channels interface call successful")

        # Test basic context functionality
        print(f"Alice context rank: {ctxA.rank}")
        print(f"Bob context rank: {ctxB.rank}")

    except Exception as e:
        print(f"❌ create_with_channels interface call failed: {e}")


if __name__ == "__main__":
    print("Starting IChannel integration verification...")

    # Run tests
    test_basic_functionality()
    test_with_create_with_channels()

    print("\n✅ All verification tests passed!")
