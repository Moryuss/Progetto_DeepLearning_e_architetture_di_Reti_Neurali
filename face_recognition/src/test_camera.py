from camera import Camera
import cv2

cam = Camera(0)

while True:
    frame = cam.read()
    if frame is None:
        break

    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
