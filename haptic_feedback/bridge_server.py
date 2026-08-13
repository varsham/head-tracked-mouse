import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BRIDGE_PORT = 5005

_state = {"hovering": False, "last_update": 0.0}
_lock = threading.Lock()


class HoverHandler(BaseHTTPRequestHandler):
	def do_POST(self):
		length = int(self.headers.get("Content-Length", 0))
		body = self.rfile.read(length)

		try:
			data = json.loads(body)
			with _lock:
				_state["hovering"] = bool(data.get("hovering", False))
				_state["last_update"] = time.time()
		except (json.JSONDecodeError, ValueError):
			pass

		self.send_response(204)
		self.send_header("Access-Control-Allow-Origin", "*")
		self.end_headers()

	def do_OPTIONS(self):
		self.send_response(204)
		self.send_header("Access-Control-Allow-Origin", "*")
		self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
		self.send_header("Access-Control-Allow-Headers", "Content-Type")
		self.end_headers()

	def log_message(self, format, *args):
		pass


def start_bridge_server():
	server = ThreadingHTTPServer(("127.0.0.1", BRIDGE_PORT), HoverHandler)
	thread = threading.Thread(target=server.serve_forever, daemon=True)
	thread.start()
	return server


def get_chrome_hover_state(max_age=0.5):
	with _lock:
		if time.time() - _state["last_update"] > max_age:
			return None
		return _state["hovering"]
