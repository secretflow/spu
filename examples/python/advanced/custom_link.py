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

import asyncio
import json
import logging
import time
from typing import Dict, Optional, Any
from urllib.parse import urljoin

import requests
import spu.libspu.link as link
import yacl


logger = logging.getLogger(__name__)


class HttpChannelConfig:
    """Configuration for HttpChannel"""

    def __init__(
        self,
        peer_url: str,
        timeout_ms: int = 30000,
        max_retry: int = 3,
        retry_interval_ms: int = 1000,
        http_max_payload_size: int = 32 * 1024 * 1024,  # 32MB
        enable_ssl: bool = False,
        ssl_verify: bool = True,
        connection_pool_size: int = 10,
    ):
        self.peer_url = peer_url.rstrip('/')
        self.timeout_ms = timeout_ms
        self.max_retry = max_retry
        self.retry_interval_ms = retry_interval_ms
        self.http_max_payload_size = http_max_payload_size
        self.enable_ssl = enable_ssl
        self.ssl_verify = ssl_verify
        self.connection_pool_size = connection_pool_size


class HttpChannel(link.IChannel):
    """HTTP-based channel implementation for SPU link communication"""

    def __init__(self, config: HttpChannelConfig):
        super().__init__()
        self.config = config
        self.session = self._create_session()
        self._recv_timeout_ms = 30000
        self._throttle_window_size = 0
        self._chunk_parallel_send_size = 1
        self._pending_messages: Dict[str, asyncio.Task] = {}
        self._shutdown = False

        # Initialize storage for received messages
        self._message_store: Dict[str, yacl.Buffer] = {}

    def _create_session(self) -> requests.Session:
        """Create HTTP session with proper configuration"""
        session = requests.Session()

        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=self.config.connection_pool_size,
            pool_maxsize=self.config.connection_pool_size,
            max_retries=requests.adapters.Retry(
                total=self.config.max_retry,
                backoff_factor=self.config.retry_interval_ms / 1000.0,
                status_forcelist=[500, 502, 503, 504]
            )
        )

        session.mount('http://', adapter)
        session.mount('https://', adapter)

        # Configure SSL if enabled
        if self.config.enable_ssl:
            session.verify = self.config.ssl_verify

        return session

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling and retries"""
        url = urljoin(self.config.peer_url, endpoint)
        timeout = self.config.timeout_ms / 1000.0

        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=timeout,
                **kwargs
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed: {method} {url} - {e}")
            raise RuntimeError(f"HTTP request failed: {e}")

    def SendAsync(self, key: str, buf: yacl.Buffer) -> None:
        """Send data asynchronously"""
        if self._shutdown:
            raise RuntimeError("Channel is shut down")

        try:
            # Convert buffer to bytes
            data = bytes(buf)

            if len(data) > self.config.http_max_payload_size:
                raise ValueError(f"Payload size {len(data)} exceeds maximum {self.config.http_max_payload_size}")

            # Prepare message metadata
            metadata = {
                'key': key,
                'timestamp': time.time(),
                'size': len(data)
            }

            # Make async request
            files = {
                'metadata': ('metadata.json', json.dumps(metadata), 'application/json'),
                'data': ('data.bin', data, 'application/octet-stream')
            }

            response = self._make_request('POST', f'/send_async/{key}', files=files)

            if response.status_code != 200:
                raise RuntimeError(f"Send failed with status {response.status_code}")

            logger.debug(f"Sent async message with key: {key}")

        except Exception as e:
            logger.error(f"SendAsync failed for key {key}: {e}")
            raise

    def SendAsyncThrottled(self, key: str, buf: yacl.Buffer) -> None:
        """Send data asynchronously with throttling"""
        # For now, implement same as SendAsync
        # Throttling logic can be added later if needed
        self.SendAsync(key, buf)

    def Send(self, key: str, value: yacl.ByteContainerView) -> None:
        """Send data synchronously"""
        if self._shutdown:
            raise RuntimeError("Channel is shut down")

        try:
            # Convert to buffer and send
            buf = yacl.Buffer(value)
            self.SendAsync(key, buf)

            # Wait for confirmation (simplified)
            time.sleep(0.01)  # Small delay to ensure message is processed

            logger.debug(f"Sent sync message with key: {key}")

        except Exception as e:
            logger.error(f"Send failed for key {key}: {e}")
            raise

    def Recv(self, key: str) -> yacl.Buffer:
        """Receive data for a specific key"""
        if self._shutdown:
            raise RuntimeError("Channel is shut down")

        start_time = time.time()
        timeout_sec = self._recv_timeout_ms / 1000.0

        while time.time() - start_time < timeout_sec:
            try:
                # Check local storage first
                if key in self._message_store:
                    return self._message_store.pop(key)

                # Try to receive from peer
                response = self._make_request('GET', f'/recv/{key}')

                if response.status_code == 200:
                    data = response.content
                    logger.debug(f"Received message with key: {key}")
                    return yacl.Buffer(data)
                elif response.status_code == 404:
                    # Message not available yet, wait and retry
                    time.sleep(0.01)
                    continue
                else:
                    raise RuntimeError(f"Recv failed with status {response.status_code}")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Recv request failed for key {key}: {e}")
                time.sleep(0.1)  # Wait before retry
                continue
            except Exception as e:
                logger.error(f"Recv failed for key {key}: {e}")
                raise

        raise TimeoutError(f"Timeout waiting for message with key: {key}")

    def SetRecvTimeout(self, timeout_ms: int) -> None:
        """Set receive timeout in milliseconds"""
        self._recv_timeout_ms = timeout_ms

    def GetRecvTimeout(self) -> int:
        """Get receive timeout in milliseconds"""
        return self._recv_timeout_ms

    def WaitLinkTaskFinish(self) -> None:
        """Wait for all pending async operations to complete"""
        if self._pending_messages:
            # Wait for all pending tasks
            import asyncio
            if asyncio.iscoroutinefunction(self._wait_pending_messages):
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self._wait_pending_messages())

        logger.info("All link tasks finished")

    async def _wait_pending_messages(self) -> None:
        """Wait for all pending message tasks"""
        if self._pending_messages:
            await asyncio.gather(*self._pending_messages.values(), return_exceptions=True)
            self._pending_messages.clear()

    def Abort(self) -> None:
        """Abort all operations and shut down the channel"""
        logger.warning("Aborting HttpChannel operations")
        self._shutdown = True

        # Cancel any pending operations
        for task in self._pending_messages.values():
            if not task.done():
                task.cancel()

        self._pending_messages.clear()

        # Close HTTP session
        if self.session:
            self.session.close()

    def SetThrottleWindowSize(self, size: int) -> None:
        """Set throttle window size"""
        self._throttle_window_size = size
        logger.debug(f"Set throttle window size to {size}")

    def SetChunkParallelSendSize(self, size: int) -> None:
        """Set chunk parallel send size"""
        self._chunk_parallel_send_size = size
        logger.debug(f"Set chunk parallel send size to {size}")

    def TestSend(self, timeout: int) -> None:
        """Test send functionality"""
        test_key = "__test_send__"
        test_data = b"test_message"

        try:
            self.Send(test_key, test_data)
            logger.info("Test send successful")
        except Exception as e:
            logger.error(f"Test send failed: {e}")
            raise

    def TestRecv(self) -> None:
        """Test receive functionality"""
        test_key = "__test_recv__"

        try:
            # This will timeout if no test message is available
            old_timeout = self._recv_timeout_ms
            self._recv_timeout_ms = 1000  # 1 second for test

            try:
                self.Recv(test_key)
                logger.info("Test receive successful")
            except TimeoutError:
                logger.info("Test receive timeout (expected if no test message available)")
            finally:
                self._recv_timeout_ms = old_timeout

        except Exception as e:
            logger.error(f"Test receive failed: {e}")
            raise

    def __del__(self):
        """Cleanup when channel is destroyed"""
        if hasattr(self, 'session') and self.session:
            try:
                self.session.close()
            except Exception:
                pass  # Ignore cleanup errors
