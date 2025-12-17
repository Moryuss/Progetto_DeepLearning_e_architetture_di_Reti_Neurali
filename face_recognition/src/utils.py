import cv2
import numpy as np


def crop_face(image, bbox, margin=0):
    """
    image: frame BGR (H, W, 3)
    bbox: (x1, y1, x2, y2)
    margin: pixel extra intorno al volto

    return: cropped face (BGR)
    """
    h, w, _ = image.shape
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    return image[y1:y2, x1:x2]


def resize_image(image, size):
    """
    image: np.ndarray
    size: (width, height)

    return: resized image
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def preprocess_for_yolo(image, input_size):
    """
    image: frame BGR
    input_size: (width, height)

    return: input tensor (1, 3, H, W)  (1 dimensione batch, 3 canali, altezza, larghezza)
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = resize_image(img, input_size)
    img = img.astype(np.float32) / 255.0

    # HWC → CHW     /height, width, channels
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)   # → NCHW

    return img


def preprocess_for_recognizer(image, input_size):
    """
    image: face crop BGR
    input_size: (width, height)

    return: input tensor (1, 3, H, W)
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = resize_image(img, input_size)
    img = img.astype(np.float32)

    # standard FaceNet / ArcFace --> trasforma da [0,255] a [-1,1]
    img = (img - 127.5) / 128.0

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img
