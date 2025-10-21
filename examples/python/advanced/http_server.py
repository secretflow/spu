#!/usr/bin/env python3
"""
Simple HTTP Server for routing HttpChannel requests
"""

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Optional
from urllib.parse import urlparse, parse_qs
import binascii


class ChannelMessageStorage:
    """Thread-safe storage for channel messages"""

    def __init__(self):
        self._storage: Dict[str, bytes] = {}
        self._lock = threading.Lock()
        self._wait_conditions: Dict[str, threading.Condition] = {}

    def store_message(self, key: str, data: bytes):
        """Store a message"""
        with self._lock:
            self._storage[key] = data
            # Notify any waiting threads
            if key in self._wait_conditions:
                self._wait_conditions[key].notify_all()

    def get_message(self, key: str, timeout_ms: int = 5000) -> Optional[bytes]:
        """Get a message, with optional timeout"""
        start_time = time.time()
        timeout_sec = timeout_ms / 1000.0

        with self._lock:
            if key in self._storage:
                return self._storage.pop(key)

            # Create condition variable for this key if not exists
            if key not in self._wait_conditions:
                self._wait_conditions[key] = threading.Condition(self._lock)

            condition = self._wait_conditions[key]

            # Wait for message with timeout
            while time.time() - start_time < timeout_sec:
                if key in self._storage:
                    return self._storage.pop(key)

                remaining_time = timeout_sec - (time.time() - start_time)
                if remaining_time > 0:
                    condition.wait(remaining_time)

            return None

    def get_stats(self):
        """Get storage statistics"""
        with self._lock:
            return {
                'total_messages': len(self._storage),
                'pending_waiters': len(self._wait_conditions),
            }


class ChannelRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for channel operations"""

    # Class-level storage shared across all instances
    storage = ChannelMessageStorage()

    def log_message(self, format, *args):
        """Override to reduce log spam"""
        print(f"[{self.address_string()}] {format % args}")

    def do_POST(self):
        """Handle POST requests"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            parsed_path = urlparse(self.path)

            if parsed_path.path == '/send':
                self._handle_send(data)
            elif parsed_path.path == '/control':
                self._handle_control(data)
            else:
                self._send_error(404, "Endpoint not found")

        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON")
        except Exception as e:
            self._send_error(500, f"Internal server error: {str(e)}")

    def do_GET(self):
        """Handle GET requests"""
        try:
            parsed_path = urlparse(self.path)
            query_params = parse_qs(parsed_path.query)

            if parsed_path.path == '/recv':
                self._handle_recv(query_params)
            elif parsed_path.path == '/stats':
                self._handle_stats()
            else:
                self._send_error(404, "Endpoint not found")

        except Exception as e:
            self._send_error(500, f"Internal server error: {str(e)}")

    def _handle_send(self, data):
        """Handle send requests"""
        action = data.get('action', '')
        key = data.get('key', '')
        data_hex = data.get('data', '')
        sender_rank = data.get('sender_rank')
        receiver_rank = data.get('receiver_rank')

        print(
            f"[Server] {action}: key={key}, from_rank={sender_rank}, to_rank={receiver_rank}, size={len(data_hex)//2}"
        )

        try:
            # Convert hex back to bytes
            message_data = binascii.unhexlify(data_hex)

            # Store message
            self.storage.store_message(key, message_data)

            response = {
                'success': True,
                'message': f'Message stored for key {key}',
                'action': action,
            }

            self._send_json_response(200, response)

        except binascii.Error as e:
            self._send_error(400, f"Invalid hex data: {str(e)}")
        except Exception as e:
            self._send_error(500, f"Failed to store message: {str(e)}")

    def _handle_recv(self, query_params):
        """Handle receive requests"""
        key = query_params.get('key', [''])[0]
        sender_rank = query_params.get('sender_rank', [''])[0]
        receiver_rank = query_params.get('receiver_rank', [''])[0]
        timeout_ms = int(query_params.get('timeout_ms', ['5000'])[0])

        print(
            f"[Server] Recv request: key={key}, from_rank={sender_rank}, to_rank={receiver_rank}, timeout={timeout_ms}"
        )

        try:
            message = self.storage.get_message(key, timeout_ms)

            if message is not None:
                response = {'found': True, 'data': message.hex(), 'key': key}
                print(f"[Server] Message found for key {key}, size={len(message)}")
            else:
                response = {'found': False, 'data': '', 'key': key}
                print(f"[Server] No message found for key {key}")

            self._send_json_response(200, response)

        except Exception as e:
            self._send_error(500, f"Failed to retrieve message: {str(e)}")

    def _handle_control(self, data):
        """Handle control requests (abort, wait, etc.)"""
        action = data.get('action', '')
        channel_id = data.get('channel_id', '')

        print(f"[Server] Control: action={action}, channel_id={channel_id}")

        # For now, just acknowledge the control request
        # In a real implementation, you might want to handle these more specifically
        response = {
            'success': True,
            'message': f'Control action {action} processed for channel {channel_id}',
            'action': action,
        }

        self._send_json_response(200, response)

    def _handle_stats(self):
        """Handle statistics requests"""
        stats = self.storage.get_stats()
        self._send_json_response(200, stats)

    def _send_json_response(self, status_code, data):
        """Send JSON response"""
        response_data = json.dumps(data).encode('utf-8')

        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_data)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        self.wfile.write(response_data)

    def _send_error(self, status_code, message):
        """Send error response"""
        error_data = {'error': True, 'message': message}
        self._send_json_response(status_code, error_data)


def run_server(port=11450):
    """Run the HTTP server"""
    server_address = ('localhost', port)
    httpd = HTTPServer(server_address, ChannelRequestHandler)

    print(f"HTTP Channel Server starting on localhost:{port}")
    print("Available endpoints:")
    print("  POST /send    - Send messages")
    print("  GET  /recv    - Receive messages")
    print("  POST /control - Control operations")
    print("  GET  /stats   - Server statistics")
    print()
    print("Press Ctrl+C to stop the server")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down...")
        httpd.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='HTTP Channel Server')
    parser.add_argument(
        '--port', type=int, default=8080, help='Server port (default: 8080)'
    )

    args = parser.parse_args()

    run_server(port=args.port)
