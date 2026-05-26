# python image_extractor.py --input perspective_transformation/sample_image.jpg --output perspective_transformation/output.jpg --width 800 --height 600

import argparse
import os
import sys
import cv2
import numpy as np

# hold clicked points
points = []


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', '-i', required=True)
	parser.add_argument('--output', '-o', default='perspective_transformation/output.jpg',)
	parser.add_argument('--width', '-W', type=int, default=600)
	parser.add_argument('--height', '-H', type=int, default=400)
	return parser.parse_args()

# records left-button clicks as corner points
def click_event(event, x, y, flags, param):
	global points
	if event == cv2.EVENT_LBUTTONDOWN:
		if len(points) < 4:
			points.append((x, y))

# orders points as top-left, top-right, bottom-right, bottom-left
def order_points(pts):

	pts = np.array(pts, dtype='float32')
	s = pts.sum(axis=1)
	diff = np.diff(pts, axis=1)
	tl = pts[np.argmin(s)]
	br = pts[np.argmax(s)]
	tr = pts[np.argmin(diff)]
	bl = pts[np.argmax(diff)]
	return np.array([tl, tr, br, bl], dtype='float32')

# transforms perspective and returns wared image
def four_point_transform(image, pts, dst_size):
	rect = order_points(pts)
	(w, h) = dst_size
	dst = np.array([
		[0, 0],
		[w - 1, 0],
		[w - 1, h - 1],
		[0, h - 1]
	], dtype='float32')

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (w, h))
	return warped

# draws a copy of img with visual markers of selected points
def draw_feedback(img, pts):
	out = img.copy()
	# draw points
	for i, p in enumerate(pts):
		cv2.circle(out, p, 5, (0, 255, 0), -1)
		cv2.putText(out, str(i + 1), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
	# draw polygon if >=2 points
	if len(pts) >= 2:
		cv2.polylines(out, [np.array(pts, dtype=np.int32)], isClosed=False, color=(255, 0, 0), thickness=1)
	return out


def main():
	global points
	args = parse_args()

	# validate input path
	if not os.path.exists(args.input):
		print(f"Input file not found: {args.input}")
		sys.exit(1)

	img = cv2.imread(args.input)
	if img is None:
		print(f"Failed to load image: {args.input}")
		sys.exit(1)

	win_name = f"{args.input}"
	cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
	cv2.setMouseCallback(win_name, click_event)

	original = img.copy()

	while True:
		# show image with feedback
		display = draw_feedback(original, points)
		cv2.imshow(win_name, display)
		key = cv2.waitKey(20) & 0xFF

		# quit in ESC
		if key == 27:
			break

		# when four points are selected, compute warp and show result
		if len(points) == 4:
			warped = four_point_transform(original, points, (args.width, args.height))
			res_win = 'S: save, ESC: discard'
			cv2.namedWindow(res_win, cv2.WINDOW_AUTOSIZE)
			cv2.imshow(res_win, warped)

			while True:
				k = cv2.waitKey(0) & 0xFF
				if k == 27:  # discard and restart
					points = []
					cv2.destroyWindow(res_win)
					break
				elif k in (ord('s'), ord('S')):
					# attempt to save
					out_path = args.output
					# if output is directory, create a filename
					if os.path.isdir(out_path):
						base = os.path.splitext(os.path.basename(args.input))[0]
						out_path = os.path.join(out_path, f"{base}_warped.jpg")

					ok = cv2.imwrite(out_path, warped)
					if ok:
						print(f'Saved result to {out_path}')
					else:
						print(f'Failed to save result to {out_path}')
					# saved then exit
					cv2.destroyWindow(res_win)
					cv2.destroyWindow(win_name)
					return

	cv2.destroyAllWindows()


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print('Interrupted')
