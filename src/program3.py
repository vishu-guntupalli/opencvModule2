'''
Custom object detector model:

Step1: sample positive images with bounding boxes
Step2: sample negative images to train the model(N >> P)
Step3: Train a Linear Support Vector Machine on your positive and negative samples
Step4: Apply hard-negative mining, this helps to reduce false positive detector
Step5: Take the false-positive samples found during the hard-negative mining stage, sort them by their confidence (i.e. probability), and re-train your classifier using these hard-negative samples
Step6: Your classifier is now trained and can be applied to your test dataset

'''

