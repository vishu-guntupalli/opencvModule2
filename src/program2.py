from __future__ import print_function
from imutils import paths
from scipy.io import loadmat
from skimage import io
import dlib
import cv2


def training():

	#trainingImagePath = './images/training/stop_sign_images'
	#trainingImageAnnotation = './images/training/stop_sign_annotations'
	#trainingOutputPath = './images/training/stop_sign_output/stop_sign_detector.svm'

	trainingImagePath = './images/training/sunflower_images'
	trainingImageAnnotation = './images/training/sunflower_annotations'
	trainingOutputPath = './images/training/sunflower_output/sunflower_detector.svm'

	options = dlib.simple_object_detector_training_options()
	images = []
	boxes = []

	# loop over the image paths
	for imagePath in paths.list_images(trainingImagePath):
		# extract the image ID from the image path and load the annotations file
		imageID = imagePath[imagePath.rfind("/") + 1:].split("_")[1]
		imageID = imageID.replace(".jpg", "")
		p = "{}/annotation_{}.mat".format(trainingImageAnnotation, imageID)
		annotations = loadmat(p)["box_coord"]

		# loop over the annotations and add each annotation to the list of bounding
		# boxes
		bb = [dlib.rectangle(left=long(x), top=long(y), right=long(w), bottom=long(h))
				for (y, h, x, w) in annotations]
		boxes.append(bb)

		# add the image to the list of images
		images.append(io.imread(imagePath))

	detector = dlib.train_simple_object_detector(images, boxes, options)
	detector.save(trainingOutputPath)
	win = dlib.image_window()
	win.set_image(detector)
	dlib.hit_enter_to_continue()


#training()

def Testing():

	#testImageFolder = "./images/training/stop_sign_testing"
	#detectorPath = "./images/training/stop_sign_output/stop_sign_detector.svm"

	testImageFolder = "./images/training/sunflower_testing"
	detectorPath = './images/training/sunflower_output/sunflower_detector.svm'

	# load the detector
	detector = dlib.simple_object_detector(detectorPath)

	# loop over the testing images
	for testingPath in paths.list_images(testImageFolder):
		# load the image and make predictions
		image = cv2.imread(testingPath)
		boxes = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

		# loop over the bounding boxes and draw them
		for b in boxes:
			(x, y, w, h) = (b.left(), b.top(), b.right(), b.bottom())
			cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

		# show the image
		cv2.imshow("Image", image)
		cv2.waitKey(0)

Testing()