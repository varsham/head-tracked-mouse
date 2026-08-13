import serial
import pyautogui
import time
from ApplicationServices import (
	AXUIElementCreateSystemWide,
	AXUIElementCopyElementAtPosition,
	AXUIElementCopyAttributeValue,
	kAXRoleAttribute,
)

PORT = "/dev/cu.usbmodem1201"

pico = serial.Serial(PORT, 115200)
system_wide = AXUIElementCreateSystemWide()

time.sleep(2)

def is_hovering_button(x, y):
	err, element = AXUIElementCopyElementAtPosition(system_wide, x, y, None)
	if err != 0 or element is None:
		print("DEBUG: position lookup failed, err =", err)
		return False

	err2, role = AXUIElementCopyAttributeValue(element, kAXRoleAttribute, None)
	print("DEBUG: role =", role, "err2 =", err2)
	return err2 == 0 and role == "AXButton"

while True:
	x, y = pyautogui.position()
	hovering = is_hovering_button(x, y)

	message = f"{x},{y},{int(hovering)}\n"
	pico.write(message.encode())

	response = pico.readline()
	if response:
		print(response.decode().strip())

	time.sleep(0.05)
