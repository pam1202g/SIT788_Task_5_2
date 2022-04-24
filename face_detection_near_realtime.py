import cv2
import time
import threading
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials

key = "02a7b436fb8b451c967537ca39ebf225"
endpoint = "https://facedetectiontask.cognitiveservices.azure.com/"
face_client = FaceClient(endpoint, CognitiveServicesCredentials(key))

left_frame = None
right_frame = None


# draw text for Face Attributes on the screen
def draw_text(frame, left, top, label):
	frame = cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
	return frame


# Age label text to be displayed on the screen
def get_age_label(face_attributes):
	return "Age: " + str(int(face_attributes.age))


# Gender label text to be displayed on the screen
def get_gender_label(face_attributes):
	gender = (face_attributes.gender.split('.'))[0]
	return "Gender: " + str(gender)


# Emotion label text to be displayed on the screen
def get_emotion_label(face_attributes):
	emotion_dict = face_attributes.emotion.__dict__
	# Removing the additional properties from the dict
	# and adding only the attributes in the emotion class returned by the API
	keys = list(emotion_dict.keys())[1:8]
	values = list(emotion_dict.values())[1:8]

	# getting the max value to extract the emotion with highest probability
	max_value = max(values)
	max_index = values.index(max_value)
	emotion = keys[max_index]
	return "Emotion: " + str(emotion)


# Coordinates for the Bounding Box
def get_coordinates(rectangle):
	left = rectangle.left
	top = rectangle.top
	right = rectangle.left + rectangle.width
	bottom = rectangle.top + rectangle.height
	return left, top, right, bottom


# Draws the Bounding Box and Face Attribute labels
def draw_result(frame, face):
	frame = cv2.putText(frame, "Bounding box & Face Attributes",
						(10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
	# Get Age from Face attributes of the face
	age = get_age_label(face.face_attributes)
	# Get Gender from Face attributes of the face
	gender = get_gender_label(face.face_attributes)

	# Get Emotion from Face attributes of the face
	emotion = get_emotion_label(face.face_attributes)
	# Get the coordinates of the rectangular bounding box
	left, top, right, bottom = get_coordinates(face.face_rectangle)
	# Draw the bounding box
	bounding_box_color = (0, 0, 255)
	frame = cv2.rectangle(frame, (left, top), (right, bottom), bounding_box_color, 3)
	# Draw the Face Attributes for Age, Gender, and Emotion
	frame = draw_text(frame, right + 50, top, age)
	frame = draw_text(frame, right + 50, top + 25, gender)
	frame = draw_text(frame, right + 50, top + 50, emotion)
	return frame


# Face Detector running in a separate thread with a lag of 1 Second
def face_detector_with_lag():
	global left_frame
	global right_frame
	attrs = ['emotion', 'age', 'gender']
	while left_frame is not None:
		time.sleep(1)
		frame = left_frame.copy()
		cv2.imwrite('tmp.jpg', frame)
		# Load the saved image to be streamed to face detection API
		img = open('tmp.jpg', "rb")
		# Pass the image to the detector webservice
		faces = face_client.face.detect_with_stream(img, return_face_attributes=attrs, detection_model='detection_01')
		# check if any face has been detected
		if len(faces) > 0:
			frame = draw_result(frame, faces[0])
			right_frame = frame


# Gets the Real-time video feed from the Webcam
def real_time_webcam_feed():
	global left_frame
	global right_frame
	webcam_feed = cv2.VideoCapture(0)
	_, first_frame = webcam_feed.read()
	first_frame = cv2.flip(first_frame, 1)
	# Copy the initial frame to both left and right frames to initialize them
	left_frame = first_frame
	right_frame = first_frame
	# Create the Face Detector thread
	face_detector = threading.Thread(target=face_detector_with_lag, args=())
	face_detector_started = False
	while True:
		_, frame = webcam_feed.read()
		frame = cv2.flip(frame, 1)
		# Copy the frame
		left_frame = frame.copy()
		# Put heading on the frame
		frame = cv2.putText(frame, "Left Frame - Video", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
		# Create a split style display
		frame_full = cv2.hconcat([frame, right_frame])
		# Show the feed in a new window
		cv2.imshow("Face detection using Azure Face API", frame_full)
		# Allow a delay of 1 Millisecond for the video to render
		cv2.waitKey(1)
		if not face_detector_started:
			face_detector_started = True
			face_detector.start()
		# Terminating condition based on window state
		if cv2.getWindowProperty("Face detection using Azure Face API", cv2.WND_PROP_VISIBLE) < 1:
			break
	# Destroy all the windows
	cv2.destroyAllWindows()


# Start webcam and capture the feed
real_time_webcam_feed()
