# Smart-Home-Security-using-Telegram-Chatbot
Using the combination of face recognition and motion detection.

OBJECTIVE

An affordable home security system using Telegram chatbot.
Uses Face recognition algorithm along with Motion detection to give correct intrusion alerts
Sends out intrusion alert with picture when unknown person enters frame.
User can request surveillance footage through commands.
The system alerts user when it can’t detect a face but there is movements.

PROBLEM STATEMENT

Smart home security is very expensive to set up.
Existing system uses motion detection alone which produces false alerts.
A subscription amount is necessary for storage of surveillance footage.
Our system uses motion detection and face recognition to filter out false results.
Surveillance footage is stored in telegram server free of cost.
Cost effective to set up.

DOWNLOAD openface_nn4.small2.v1.t7 FACE RECOGNITION FROM THE WEB.

EXTRACT THE .ZIP FILES.

CREATE FACE DATASET USING build_face_dataset.py

// USAGE>>>
python build_face_dataset.py --cascade haarcascade_frontalface_default.xml --output dataset/<dataset-name>
  
EXTRACT EMBEDDINGS FROM FACE DATASETS.

// USAGE>>>
python embeddings.py --dataset dataset \
	--embeddings output/embeddings.pickle \
	--detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7

TRAIN A MACHINE LEARNING MODEL USING THE FACE DATASETS

// USAGE>>>
python train_model.py --embeddings output/embeddings.pickle \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle
	

RUN THE HOME SECURITY SYSTEM

//USAGE>>>python Home_security_bot.py --detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7 \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle \
	
ALGORITHM - MOTION DETECTION

Input the video frames

Initialize the first frame to a still frame with no motion.

Resize Convert frames to grayscale, apply Gaussian blur.

	frame = imutils.resize(frame, width =500) 

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

	gray = cv2.GaussianBlur(gray, (21, 21), 0) 
	
Compute difference between first frame and subsequent  frames from video stream

Take the absolute value of their corresponding pixel intensity differences 

delta = |background model – current frame| 

	frameDelta = cv2.absdiff(firstFrame, gray)
	
	thresh = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)[1]
	
Start looping over each contour, filter small ones.

If contour area is larger than minimum area, draw bounding box surrounding the foreground and motion region.

	if cv2.contourArea(c) < args(“min area”): 

		(x, y, w, h) = cv2.boundingRect(c) 

		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
		
		
ALGORITHM - FACE RECOGNITION


Gather the face dataset of the users.

Each input batch of data includes positive image and negative image.

Extract embeddings from face dataset

	Loading Embedder
	
	embedder = cv2.dnn.readNetFromTorch(args["embedding model"]) 
	
Detect faces in the image by passing through the detector network.

	image = cv2.imread(args["image"]) 

	image = imutils.resize(image, width=600) 

	(h, w) = image.shape[:2] 

	imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 	1.0, (300, 	300),(104.0, 177.0, 123.0), swapRB=False, crop=False) 

Extract the detection with the highest confidence 

Use confidence to filter out weak detections.

	for i in range(0, detections.shape[2]): 
	
		confidence = detections[0, 0, i, 2] 

	if confidence > args["confidence"]:
	
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) 
		
		(startX, startY, endX, endY) = box.astype("int") 
		
		face = image[startY:endY, startX:endX] 
		
		(fH, fW) = face.shape[:2] 
		
RECORDING VIDEO FOOTAGE

       footage = cv2.VideoWriter('footage.avi',cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480)) 

footage is file to where the video is written.

XVID is codec used for encoding the video.

30 is frame rate of the footage.

640x480 is the resolution at which the video will be recorded.

	frame = vs.read() 

	footage.write(frame)  
	
TELEGRAM NOTIFICATIONS

When movement is detected, the system also looks for the face if it is unknown image is captured and send to user.

	if(le.classes_[j] == "unknown" and motion_detected):
	
                cv2.imwrite(filename= '00.jpg', img=frame) 
			  
		cv2.imwrite('00.jpg', img=frame) 
		
		telegram_bot.sendPhoto(chat_id=562320888,photo=open(‘path’, 'rb'), caption='Motion detected: Unknown person') 
		
	if command == 'Footage’: 
	
		telegram_bot.sendMessage (chat_id, str("Sending surveillance footage..")) 
	
		telegram_bot.sendDocument(chat_id, open(“Footage.avi” , ‘rb’) ) 
		
Recording a 5 second clip:

	if command == 'Video' : 
	
		tim=1 
		
		cont = cont + 1 
		
		start_time = time.time() 
		
		print(int(start_time)) 
		
		while(int(time.time() - start_time) < 5): 
		
		if(tim==1): 
		
			result.write(frame) 
			
			telegram_bot.sendMessage (chat_id, str("Recorded! 5 seconds added to video!")) 
			
			telegram_bot.sendDocument(chat_id, document=open('filename.avi’, ‘rb’)) 
			
RESULT AND CONCLUSION

The incorporation of motion detection and face recognition has helped filter out false alerts to an extend.
Telegram being open source and free allows for storage of video files and images sent to users on their servers.
OpenCV is fast and less resource intensive and gives good frame rate.
Face recognition is done with OpenCV and is fairly accurate.
Face recognition only works when someone is facing the camera.
Face recognition working side to side with motion detection helps to detect intruders even if face can’t be detected. 


FUTURE SCOPE

Object detection can be implemented completely using TensorFlow.
TensorFlow can detect a person as a whole and not by just facial features.
But TensorFlow alone is resource heavy, so it will work more efficiently with OpenCV.
A more powerful hardware with more graphical performance can easily run Smart surveillance using TensorFlow.
Use of IR camera can make the system usable in low-light.

			



		
		

































































































































































Most of the code is used from https://www.pyimagesearch.com/, Thanks to Adrian from PyImageSearch!
We combined two separate algorithms and added our own to make this system work.





