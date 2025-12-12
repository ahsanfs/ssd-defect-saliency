import cv2
import numpy as np

drawing = False
ix, iy = -1, -1
mask = None

def draw(event, x, y, flags, param):
    global ix, iy, drawing, mask

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(mask, (ix, iy), (x, y), 255, 5)
            cv2.line(img_show, (ix, iy), (x, y), (0,0,255), 2)
            ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(mask, (ix, iy), (x, y), 255, 5)

# -------- Main ----------
img_path = "./ssd_frames/ssd_17.jpg"
img = cv2.imread(img_path)
h, w = img.shape[:2]

img_show = img.copy()
mask = np.zeros((h, w), dtype=np.uint8)

cv2.namedWindow("Annotate Scratch")
cv2.setMouseCallback("Annotate Scratch", draw)

while True:
    cv2.imshow("Annotate Scratch", img_show)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        cv2.imwrite(img_path.replace(".jpg","_mask.png"), mask)
        print("Mask saved!")
        break
    if key == 27:
        break

cv2.destroyAllWindows()
