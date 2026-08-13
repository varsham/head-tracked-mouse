import pyautogui
from ApplicationServices import (
	AXUIElementCreateSystemWide,
	AXUIElementCopyElementAtPosition,
	AXUIElementCopyAttributeValue,
	kAXRoleAttribute,
)

system_wide = AXUIElementCreateSystemWide()

def element_role_at_mouse():
	x, y = pyautogui.position()
	err, element = AXUIElementCopyElementAtPosition(system_wide, x, y, None)
	if err != 0 or element is None:
		return None

	err2, role = AXUIElementCopyAttributeValue(element, kAXRoleAttribute, None)
	if err2 != 0:
		return None

	return role

while True:
	role = element_role_at_mouse()
	if role == "AXButton":
		print("Hovering over a button")
