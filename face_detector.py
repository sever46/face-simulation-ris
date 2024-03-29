# the face detector interface
# detect_faces(img) -> Array
#   will return an array of detected faces (described later)
# clear_face_cache() will remove all known faces
# face_cache_length() number of known faces
# detected face is a structure which looks like
# { is_new : Bool, box : Rect(Tuple) }
# rect is (x1, y1, x2, y2), does not contain w,h

class FaceDetector:
    def __init__(self) -> None:
        self.face_cache = []
    
    def clear_face_cache(self):
        self.face_cache = []

    def face_cache_length(self):
        return len(self.face_cache)
    
    def detect_faces(self, img):
        # TODO implement
        return [
            {"is_new": True, "box": (10, 10, 60, 60)},
            {"is_new": False, "box": (100, 100, 150, 150)}
        ]