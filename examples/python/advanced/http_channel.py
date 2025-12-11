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

import spu.libspu.link as link
import requests
import time
from typing import Optional
from urllib.parse import urljoin


class HttpChannel(link.IChannel):
    """HTTP-based channel implementation for IChannel interface"""

    def __init__(
        self,
        local_rank: int,
        remote_rank: int,
        base_port: int = 8080,
        channel_id: Optional[str] = None,
    ):
        """
        Initialize HTTP Channel

        Args:
            local_rank: Local node rank
            remote_rank: Remote node rank
            base_port: Port of the HTTP server
            channel_id: Unique channel identifier (optional)
        """
        super().__init__()
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
        self._session.headers.update(
            {
                'Content-Type': 'application/json',
                'X-Channel-Id': self.channel_id,
                'X-Local-Rank': str(local_rank),
                'X-Remote-Rank': str(remote_rank),
            }
        )

    def _send_key(self, key: str) -> str:
        """Create a unique key for sending messages"""
        return f"{key}_{self.local_rank}_{self.remote_rank}"

    def _recv_key(self, key: str) -> str:
        """Create the key for receiving messages"""
        return f"{key}_{self.remote_rank}_{self.local_rank}"

    def _send_request(
        self,
        path: str,
        data: Optional[dict] = None,
        method: str = 'POST',
        params: Optional[dict] = None,
    ) -> Optional[requests.Response]:
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
            print(f"[{self.channel_id}] HTTP request failed: {e}")
            return None

    def SendAsync(self, key: str, buf: bytes) -> None:
        """Asynchronously send data via HTTP"""
        final_key = self._send_key(key)
        print(f"[{self.channel_id}] SendAsync: key={final_key}, size={len(buf)}")

        data = {
            'action': 'send_async',
            'key': final_key,
            'data': buf.hex(),  # Convert bytes to hex for JSON transport
            'sender_rank': self.local_rank,
            'receiver_rank': self.remote_rank,
        }

        self._send_request('/send', data)

    def SendAsyncThrottled(self, key: str, buf: bytes) -> None:
        """Asynchronously send data with throttling via HTTP"""
        final_key = self._send_key(key)
        print(
            f"[{self.channel_id}] SendAsyncThrottled: key={final_key}, size={len(buf)}"
        )

        data = {
            'action': 'send_async_throttled',
            'key': final_key,
            'data': buf.hex(),
            'sender_rank': self.local_rank,
            'receiver_rank': self.remote_rank,
            'throttle_window_size': self.throttle_window_size,
        }

        self._send_request('/send', data)

    def Send(self, key: str, value: bytes) -> None:
        """Synchronously send data via HTTP"""
        final_key = self._send_key(key)
        print(f"[{self.channel_id}] Send: key={final_key}, size={len(value)}")

        data = {
            'action': 'send',
            'key': final_key,
            'data': value.hex(),
            'sender_rank': self.local_rank,
            'receiver_rank': self.remote_rank,
        }

        self._send_request('/send', data)

    def Recv(self, key: str) -> bytes:
        """Receive data via HTTP with retry mechanism"""
        final_key = self._recv_key(key)
        print(f"[{self.channel_id}] Recv: key={final_key}")

        max_retries = 5

        for attempt in range(max_retries + 1):  # +1 for the initial attempt
            params = {
                'key': final_key,
                'sender_rank': self.remote_rank,
                'receiver_rank': self.local_rank,
                'timeout_ms': self.recv_timeout,
            }

            response = self._send_request('/recv', method='GET', params=params)
            if response:
                result = response.json()
                if result.get('found', False):
                    data_hex = result.get('data', '')
                    if attempt > 0:
                        print(
                            f"[{self.channel_id}] Recv SUCCESS on retry {attempt}: key={final_key}, size={len(data_hex)//2}"
                        )
                    return bytes.fromhex(data_hex)

            if attempt < max_retries:
                retry_delay = 0.05 * (2**attempt)  # 50ms, 100ms, 200ms, 400ms, 800ms
                retry_delay = min(retry_delay, 1.0)  # 最大等待1秒
                print(
                    f"[{self.channel_id}] Recv RETRY {attempt+1}/{max_retries}: key={final_key}, timeout={self.recv_timeout}ms, delay={retry_delay*1000:.0f}ms"
                )
                time.sleep(retry_delay)

        print(
            f"[{self.channel_id}] Recv FAILED after {max_retries + 1} attempts: key={final_key}"
        )
        return b""

    def SetRecvTimeout(self, timeout_ms: int) -> None:
        """Set receive timeout"""
        print(f"[{self.channel_id}] SetRecvTimeout: {timeout_ms}ms")
        self.recv_timeout = timeout_ms

    def GetRecvTimeout(self) -> int:
        """Get receive timeout"""
        return self.recv_timeout

    def WaitLinkTaskFinish(self) -> None:
        """Wait for link tasks to finish"""
        print(f"[{self.channel_id}] WaitLinkTaskFinish")

        # Send wait request to server
        data = {'action': 'wait_link_task_finish', 'channel_id': self.channel_id}
        self._send_request('/control', data)

    def Abort(self) -> None:
        """Abort operation"""
        print(f"[{self.channel_id}] Abort")

        # Send abort request to server
        data = {'action': 'abort', 'channel_id': self.channel_id}
        self._send_request('/control', data)

    def SetThrottleWindowSize(self, size: int) -> None:
        """Set throttle window size"""
        print(f"[{self.channel_id}] SetThrottleWindowSize: {size}")
        self.throttle_window_size = size

    def TestSend(self, timeout: int) -> None:
        """Test send functionality"""
        print(f"[{self.channel_id}] TestSend with timeout={timeout}")
        self.Send("test", b"")

    def TestRecv(self) -> None:
        """Test receive functionality"""
        print(f"[{self.channel_id}] TestRecv")
        self.Recv("test")

    def SetChunkParallelSendSize(self, size: int) -> None:
        """Set chunk parallel send size"""
        print(f"[{self.channel_id}] SetChunkParallelSendSize: {size}")
        self.chunk_parallel_send_size = size

    def __del__(self):
        """Close the channel and clean up resources"""
        try:
            if hasattr(self, '_session'):
                self._session.close()
            print(f"[{self.channel_id}] Channel closed")
        except Exception:
            pass
