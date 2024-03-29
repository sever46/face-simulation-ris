#import cv2
#import tkinter as t
#
#class VideoOverviewWindow:
#    # constructor
#    def __init__(self, video, face_detector) -> None:
#        self.root = t.Tk()
#        self.root.title = "Video overview"
#        self.video = video
#        self.current_frame = 0
#        self.face_detector = face_detector
#        frame = t.Frame(self.root, width=400, height=300)
#        frame.grid(column=0, row=0, padx=10, pady=10)
#        t.Label(frame, text="Frame Preview").grid(row=0, column=0, padx=5, pady=5)
#
#    def start(self):
#        self.root.mainloop()

import tkinter as t
from tkinter import ttk
from PIL import Image, ImageTk
import argparse
import cv2
import sys
import importlib.util

class VideoOverviewWindow:
    def __init__(self, video, face_detector) -> None:
        self.root = t.Tk()
        self.root.title("Video Test")
        self.video = video
        self.face_detector = face_detector
        self.current_frame_number = 0
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        self.frame = t.Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        t.Label(self.frame, text="Preview").pack()

        self.image_label = t.Label(self.frame)
        self.image_label.pack()

        control_frame = t.Frame(self.root)
        control_frame.pack(pady=10)

        self.frame_entry = t.Entry(control_frame, width=10)
        self.frame_entry.pack(side=t.LEFT, padx=5)
        
        t.Button(control_frame, text="Go", command=self.go_to_frame).pack(side=t.LEFT)
        t.Button(control_frame, text="Previous", command=self.previous_frame).pack(side=t.LEFT)
        t.Button(control_frame, text="Next", command=self.next_frame).pack(side=t.LEFT)
        t.Button(control_frame, text="Clear Face Cache", command=self.clear_face_cache).pack(side=t.LEFT)
        self.root.bind("<Left>", lambda event: self.previous_frame())
        self.root.bind("<Right>", lambda event: self.next_frame())

        self.log = t.Text(self.root, state="disabled", height=5, wrap="word")
        self.log.pack(padx=10, pady=10, fill=t.BOTH, expand=True)

        self.update_image()

    def log_message(self, message):
        self.log.configure(state="normal")
        self.log.insert(t.END, message + "\n")
        self.log.configure(state="disabled")
        self.log.see(t.END)

    def clear_face_cache(self):
        self.face_detector.clear_face_cache()
        self.update_image()
        self.log_message("cleared face cache.")

    def update_image(self):
        self.frame_entry.delete(0, t.END)
        self.frame_entry.insert(0, str(self.current_frame_number))
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        ret, frame = self.video.read()
        if not ret:
            self.log_message("Failed to get the frame.")
            return
        faces = self.face_detector.detect_faces(frame)
        newfaces = 0
        for face in faces:
            x1, y1, x2, y2 = face["box"]
            if face["is_new"]:
                newfaces += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        if newfaces > 0:
            self.log_message(f"{newfaces} new faces detected in frame {self.current_frame_number}. face cache: {self.face_detector.face_cache_length()}")
        
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_label.imgtk = imgtk
        self.image_label.configure(image=imgtk)

    def go_to_frame(self):
        frame_number = int(self.frame_entry.get())
        if frame_number >= self.total_frames:
            frame_number = self.total_frames - 1
        if frame_number >= 0:
            self.current_frame_number = frame_number
            self.update_image()
        else:
            self.log_message("out of range")

    def next_frame(self):
        # Check the total number of frames in the video
        # Prevent going beyond the last frame
        if self.current_frame_number < self.total_frames - 1:
            self.current_frame_number += 1
            self.update_image()
        else:
            self.log_message("Reached the end of the video.")

    def previous_frame(self):
        # Prevent going before the first frame
        if self.current_frame_number > 0:
            self.current_frame_number -= 1
            self.update_image()
        else:
            self.log_message("Already at the first frame.")

    def start(self):
        self.root.mainloop()


def parse_arguments_and_construct_class():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Face detection simulator")
    # Add arguments
    parser.add_argument("--video", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--detector", type=str, required=True, help="Path to the Python file containing the detector.")
    parser.add_argument("--class", type=str, required=True, dest="class_name", help="Name of the class in the Python file.")
    # Parse arguments
    args = parser.parse_args()
    video_path = args.video
    detector_path = args.detector
    class_name = args.class_name
    # Import the specified Python file
    try:
        spec = importlib.util.spec_from_file_location("module.name", detector_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Error importing module from {detector_path}: {e}")
        sys.exit(1)
    # Construct the specified class
    try:
        class_ = getattr(module, class_name)
        class_instance = class_()
    except AttributeError:
        print(f"Class {class_name} not found in the module.")
        sys.exit(1)
    except Exception as e:
        print(f"Error constructing the class {class_name}: {e}")
        sys.exit(1)
    # Return the video path and the constructed class instance
    return (video_path, class_instance)

if __name__ == "__main__":
    vpath, cinst = parse_arguments_and_construct_class()
    cap = cv2.VideoCapture(vpath)
    VideoOverviewWindow(cap, cinst).start()


