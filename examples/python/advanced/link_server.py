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


from asyncio.log import logger
import threading
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class SendRequest(BaseModel):
    data: str  # Base64 encoded data


class MailBox:
    def __init__(self):
        self._msgbox: dict = {}
        self._cond = threading.Condition()

    def onSent(self, frm: int, key: str, data: Any) -> None:
        """Called when a key is sent to self"""
        with self._cond:
            mkey = (frm, key)
            assert mkey not in self._msgbox, f"{mkey} exist {self._msgbox.keys()}"
            self._msgbox[mkey] = data
            self._cond.notify_all()

    def recv(self, frm: int, key: str, timeout=None) -> Any:
        """Wait until the key is set, returns the value"""
        # print(f"recv {key}: {sender_rank} -> {self.rank}")
        mkey = (frm, key)
        with self._cond:
            # Wait until message arrives, then consume it
            notified = self._cond.wait_for(
                lambda: mkey in self._msgbox, timeout=timeout
            )
            if notified:
                return self._msgbox.pop(mkey)
            else:
                raise TimeoutError(f"timeout waiting for {mkey}")


_sessions: dict[str, MailBox] = {}


def create_session(session_name: str) -> MailBox:
    if session_name in _sessions:
        raise ValueError(f"Session already exists: session={session_name}")

    mailbox = MailBox()
    _sessions[session_name] = mailbox
    return mailbox


def delete_session(session_name: str) -> None:
    if session_name in _sessions:
        del _sessions[session_name]


@app.put("/sessions/{session_name}/msg/{key}/from/{from_rank}")
def send_msg(
    session_name: str, key: str, from_rank: int, request: SendRequest
) -> dict[str, str]:
    # Handle the incoming message
    if session_name not in _sessions:
        logger.error(f"Session not found: session={session_name}")
        raise HTTPException(status_code=404, detail="Session not found")

    # The receiver rank should be the rank of the server hosting this endpoint
    # We don't need to validate to_rank since the request is coming to this server

    # Use the proper onSent mechanism from CommunicatorBase
    _sessions[session_name].onSent(from_rank, key, request.data)
    return {"status": "ok"}
