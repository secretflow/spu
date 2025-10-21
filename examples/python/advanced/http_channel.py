#!/usr/bin/env python3
"""
HTTP Channel implementation for IChannel interface
"""

import spu.libspu.link as link
import requests
import json
import time
from typing import Optional
from urllib.parse import urljoin


class HttpChannel(link.IChannel):
    """HTTP-based channel implementation for IChannel interface"""

    def __init__(
        self, 
        name: str, 
        local_rank: int, 
        remote_rank: int,
        base_port: int = 8080,
        channel_id: Optional[str] = None
    ):
        """
        Initialize HTTP Channel
        
        Args:
            name: Channel name for identification
            local_rank: Local node rank
            remote_rank: Remote node rank  
            base_port: Port of the HTTP server
            channel_id: Unique channel identifier (optional)
        """
        super().__init__()
        self.name = name
        self.local_rank = local_rank
        self.remote_rank = remote_rank
        self.base_port = base_port
        self.base_url = f"http://localhost:{base_port}"
        self.channel_id = channel_id or f"channel_{local_rank}_{remote_rank}"
        
        # Configuration settings
        self.recv_timeout = 5000  # 5 seconds default
        self.throttle_window_size = 1024
        self.chunk_parallel_send_size = 4
        
        # Session for HTTP requests
        self._session = requests.Session()
        self._session.headers.update({
            'Content-Type': 'application/json',
            'X-Channel-Id': self.channel_id,
            'X-Local-Rank': str(local_rank),
            'X-Remote-Rank': str(remote_rank)
        })

    def _send_key(self, key: str) -> str:
        """Create a unique key for sending messages"""
        return f"{key}_{self.local_rank}_{self.remote_rank}"

    def _recv_key(self, key: str) -> str:
        """Create the key for receiving messages"""
        return f"{key}_{self.remote_rank}_{self.local_rank}"

    def _send_request(self, path: str, data: Optional[dict] = None, method: str = 'POST', params: Optional[dict] = None) -> Optional[requests.Response]:
        """Send HTTP request to server"""
        try:
            url = urljoin(self.base_url, path)
            if method.upper() == 'GET':
                response = self._session.get(url, params=params, timeout=10)
            elif method.upper() == 'POST':
                response = self._session.post(url, json=data, params=params, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            response.raise_for_status()
            return response
        except Exception as e:
            print(f"[{self.name}] HTTP request failed: {e}")
            return None

    def SendAsync(self, key: str, buf: bytes) -> None:
        """Asynchronously send data via HTTP"""
        final_key = self._send_key(key)
        print(f"[{self.name}] SendAsync: key={final_key}, size={len(buf)}")

        data = {
            'action': 'send_async',
            'key': final_key,
            'data': buf.hex(),  # Convert bytes to hex for JSON transport
            'sender_rank': self.local_rank,
            'receiver_rank': self.remote_rank
        }

        self._send_request('/send', data)

    def SendAsyncThrottled(self, key: str, buf: bytes) -> None:
        """Asynchronously send data with throttling via HTTP"""
        final_key = self._send_key(key)
        print(f"[{self.name}] SendAsyncThrottled: key={final_key}, size={len(buf)}")

        data = {
            'action': 'send_async_throttled',
            'key': final_key,
            'data': buf.hex(),
            'sender_rank': self.local_rank,
            'receiver_rank': self.remote_rank,
            'throttle_window_size': self.throttle_window_size
        }

        self._send_request('/send', data)

    def Send(self, key: str, value: bytes) -> None:
        """Synchronously send data via HTTP"""
        final_key = self._send_key(key)
        print(f"[{self.name}] Send: key={final_key}, size={len(value)}")

        data = {
            'action': 'send',
            'key': final_key,
            'data': value.hex(),
            'sender_rank': self.local_rank,
            'receiver_rank': self.remote_rank
        }

        self._send_request('/send', data)

    def Recv(self, key: str) -> bytes:
        """Receive data via HTTP"""
        final_key = self._recv_key(key)
        print(f"[{self.name}] Recv: key={final_key}")

        params = {
            'key': final_key,
            'sender_rank': self.remote_rank,
            'receiver_rank': self.local_rank,
            'timeout_ms': self.recv_timeout
        }

        response = self._send_request('/recv', method='GET', params=params)
        if response:
            result = response.json()
            if result.get('found', False):
                data_hex = result.get('data', '')
                return bytes.fromhex(data_hex)

        return b""

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

        # Send wait request to server
        data = {
            'action': 'wait_link_task_finish',
            'channel_id': self.channel_id
        }
        self._send_request('/control', data)

    def Abort(self) -> None:
        """Abort operation"""
        print(f"[{self.name}] Abort")

        # Send abort request to server
        data = {
            'action': 'abort',
            'channel_id': self.channel_id
        }
        self._send_request('/control', data)

    def SetThrottleWindowSize(self, size: int) -> None:
        """Set throttle window size"""
        print(f"[{self.name}] SetThrottleWindowSize: {size}")
        self.throttle_window_size = size

    def TestSend(self, timeout: int) -> None:
        """Test send functionality"""
        print(f"[{self.name}] TestSend with timeout={timeout}")
        self.Send("test", b"")

    def TestRecv(self) -> None:
        """Test receive functionality"""
        print(f"[{self.name}] TestRecv")
        self.Recv("test")

    def SetChunkParallelSendSize(self, size: int) -> None:
        """Set chunk parallel send size"""
        print(f"[{self.name}] SetChunkParallelSendSize: {size}")
        self.chunk_parallel_send_size = size

    def __del__(self):
        """Close the channel and clean up resources"""
        try:
            if hasattr(self, '_session'):
                self._session.close()
            print(f"[{self.name}] Channel closed")
        except Exception:
            pass