import sys

while True:
	mouse_pos = sys.stdin.readline()

	if mouse_pos:
		mouse_pos = mouse_pos.strip()
		x_pos, y_pos, hover_pos = mouse_pos.split(",")
		x = int(x_pos.strip())
		y = int(y_pos.strip())
		hovering = int(hover_pos.strip())

		if (x == 0 or x == 1727):
			print("boundary!")
		elif (y == 0 or y == 1116):
			print("boundary!")
		elif hovering:
			print("BUZZ")
		else:
			print("x:", x, "y:", y)
