# the face detector interface
# detect_faces(img) -> Array
#   will return an array of detected faces (described later)
# clear_face_cache() will remove all known faces
# face_cache_length() number of known faces
# detected face is a structure which looks like
# { is_new : Bool, box : Rect(Tuple) }
# rect is (x1, y1, x2, y2), does not contain w,h

from ultralytics import YOLO
import cv2

FLAN_IDX = 1

class FaceDetector:
    def __init__(self) -> None:
        self.face_cache = []
        self.model = YOLO("yolov8n.pt")
        self.sift = cv2.SIFT_create()
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=FLAN_IDX, trees=5), dict(checks=50))
    
    def clear_face_cache(self):
        self.face_cache = []

    def face_cache_length(self):
        return len(self.face_cache)

    def _is_match(self, descriptors1, descriptors2, ratio1=0.96, ratio2=0.5):
        """
        Determine if two sets of descriptors match.
        """
        matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)
        # Apply ratio test
        good_matches = [m for m, n in matches if m.distance < ratio1 * n.distance]
        return len(good_matches) > (len(matches) * ratio2)  # Threshold for a good match, can be adjusted

    
    def check_face(self, image):
        gray_face = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray_face, None)
        for _, cached_descriptors in self.face_cache:
            if self._is_match(descriptors, cached_descriptors):
                return True  # Face exists in cache
        self.face_cache.append((keypoints, descriptors))  # Add new face to cache
        return False
        

    def detect_faces(self, img):
        # TODO implement
        result = []
        res = self.model.predict(img, imgsz=(256, 320), show=False, verbose=False, classes=[0], device="")
        for r in res:
            bbox = r.boxes.xyxy
            if bbox.nelement() == 0:
                continue
            bbox = bbox[0]
            coor = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            x1, y1, x2, y2 = coor
            cropped_img = img[y1:y2, x1:x2]
            result.append({"is_new": not self.check_face(cropped_img), "box": coor})
        return result
        return [
            {"is_new": True, "box": (10, 10, 60, 60)},
            {"is_new": False, "box": (100, 100, 150, 150)}
        ]