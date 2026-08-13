import time

from bridge_server import start_bridge_server, get_chrome_hover_state

start_bridge_server()
print("Bridge server listening on http://127.0.0.1:5005 — hover over things in Chrome...")

last = None
while True:
	state = get_chrome_hover_state()
	if state != last:
		last = state
		print("hovering:", state)

	time.sleep(0.05)
