
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time, datetime
import telepot
import telegram
import os
from telepot.loop import MessageLoop
import cv2
import threading
import sys
import site




def action(msg):
    chat_id = msg['chat']['id']
    command = msg['text']
    tim=0
    cont=0


    print( 'Received: %s' % command)
    if command == 'Video' :
        tim=1
        cont = cont + 1
        start_time = time.time()
        print(int(start_time))

        while(int(time.time() - start_time) < 5):
            if(tim==1):
                result.write(frame)

        print("Done recording")
        telegram_bot.sendMessage (chat_id, str("Recorded! 5 second clip is being sent..."))
        telegram_bot.sendDocument(chat_id, document=open('/home/sanjay/project/opencv-face/filename.avi', 'rb'))
    elif command == 'Snap':
        
        telegram_bot.sendPhoto(chat_id=chat_id, photo=open('/home/sanjay/project/opencv-face/04.jpg', 'rb'), caption= 'Last motion detected Frame')
        






    elif command == 'Footage':
        telegram_bot.sendMessage (chat_id, str("Sending surveillance footage..\nCan take some time depending on file size and internet connectivity."))
        telegram_bot.sendDocument(chat_id, document=open('/home/sanjay/project/opencv-face/footage.avi', 'rb'))




    elif command == 'Photo':
        cv2.imwrite(filename='01.jpg', img=frame)
        telegram_bot.sendPhoto(chat_id=chat_id, photo=open('/home/sanjay/project/opencv-face/01.jpg', 'rb'))



    elif command == 'Status':
        if motion_detected:
            cv2.imwrite("02.jpg", img=frame)
            telegram_bot.sendPhoto(chat_id, photo=open('/home/sanjay/project/opencv-face/02.jpg', 'rb'))
            telegram_bot.sendMessage (chat_id, str("There is movement! To Record a 5 second clip, send COMMAND 'Video'!"))
        else:

            telegram_bot.sendMessage(chat_id, str('No movements detected..House is Secure!'))



    else:
        telegram_bot.sendMessage(chat_id, str("Wrong Command! Valid commands are: \n\n1) Video   --Records a 5 sec video clip \n\n2) Photo  --takes a snapshot of surveillance \n\n3) Status --security check \n\n4) Footage  --Returns Surveillance footage \n\n5) Snap   --Last image captured when movement was detected"))




telegram_bot = telepot.Bot('1008383598:AAEffUU-y3LH5UywhlrKgUXnumDkRQtf-CU')
print (telegram_bot.getMe())

MessageLoop(telegram_bot, action).run_as_thread()
print( 'Chatbot is Online!')





ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")


ap.add_argument("-c", "--confidence", type=float, default=0.49,
	help="minimum probability to filter weak detections")

ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")

args = vars(ap.parse_args())


print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])


recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
firstFrame = None

fps = FPS().start()
l=0
k=0
j=0



result = cv2.VideoWriter('filename.avi',
						cv2.VideoWriter_fourcc(*'XVID'),
						30, (640, 480))
footage = cv2.VideoWriter('footage.avi',
						cv2.VideoWriter_fourcc(*'XVID'),
						30, (640, 480))











l=0
mot =1
vid =0
countr=0
strt = time.time()




while True:
	
	frame = vs.read()
	footage.write(frame)
	motion_detected = False







	frame = frame if args.get("video", None) is None else frame[1]
	text = "Unoccupied"
	



	if frame is None:
		break

	
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)







	if firstFrame is None:
		firstFrame = gray
		continue



	
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)[1]

	
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	




	
	for c in cnts:



            if cv2.contourArea(c) < args["min_area"]:
                continue
            (x, y, b, s) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + b, y + s), (0, 255, 0), 2)
            text = "Occupied"
            motion_detected = True



	
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)











	
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	
	detector.setInput(imageBlob)
	detections = detector.forward()

	
	for i in range(0, detections.shape[2]):

		
		confidence = detections[0, 0, i, 2]

		
		if confidence > args["confidence"]:
			
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]


			
			if fW < 20 or fH < 20:
				continue

			
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]




			
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

			
			
			
                          
                            
			
			
                       



			
			if(le.classes_[j] == "unknown" and motion_detected):

                             cv2.imwrite(filename= '00.jpg', img=frame)
                             l=l+1
                           
                             if l==20:
                                 cv2.imwrite('00.jpg', img=frame)
                                 l=0
                                 telegram_bot.sendPhoto(chat_id=562320888, photo=open('/home/sanjay/project/opencv-face/00.jpg', 'rb'), caption='Motion detected: Unknown person')
                                 telegram_bot.sendPhoto(chat_id=495615272, photo=open('/home/sanjay/project/opencv-face/00.jpg', 'rb'), caption='Motion detected: Unknown person')
                                 telegram_bot.sendPhoto(chat_id=374931237, photo=open('/home/sanjay/project/opencv-face/00.jpg', 'rb'), caption='Motion detected: Unknown person')
                                 telegram_bot.sendPhoto(chat_id=649640936, photo=open('/home/sanjay/project/opencv-face/00.jpg', 'rb'), caption='Motion detected: Unknown person')






                           









                             break










	
	elsp = int(time.time() - strt)

	if motion_detected:
            cv2.imwrite("04.jpg", img=frame)
            
            countr=countr+1
            if countr== 40:
                cv2.imwrite("04.jpg", img=frame)
                telegram_bot.sendPhoto(chat_id=562320888, photo=open('/home/sanjay/project/opencv-face/04.jpg', 'rb'), caption='There is movement! send Footage to get surveillance footage or Video to record 5s clip!')
                telegram_bot.sendPhoto(chat_id=495615272, photo=open('/home/sanjay/project/opencv-face/04.jpg', 'rb'), caption='There is movement! send Footage to get surveillance footage or Video to record 5s clip!')
                telegram_bot.sendPhoto(chat_id=374931237, photo=open('/home/sanjay/project/opencv-face/04.jpg', 'rb'), caption='There is movement! send Footage to get surveillance footage or Video to record 5s clip!')
                telegram_bot.sendPhoto(chat_id=649640936, photo=open('/home/sanjay/project/opencv-face/04.jpg', 'rb'), caption='There is movement! send Footage to get surveillance footage or Video to record 5s clip!')
                
                
            if countr== 100:
                cv2.imwrite("04.jpg", img=frame)
                telegram_bot.sendPhoto(chat_id=562320888, photo=open('/home/sanjay/project/opencv-face/04.jpg', 'rb'), caption='There is movement! send Footage to get surveillance footage or Video to record 5s clip!')
                
            if countr== 180:
                cv2.imwrite("04.jpg", img=frame)
                telegram_bot.sendPhoto(chat_id=562320888, photo=open('/home/sanjay/project/opencv-face/04.jpg', 'rb'), caption='There is movement! send Footage to get surveillance footage or Video to record 5s clip!')
                
            if countr== 280:
                cv2.imwrite("04.jpg", img=frame)
                telegram_bot.sendPhoto(chat_id=562320888, photo=open('/home/sanjay/project/opencv-face/04.jpg', 'rb'), caption='There is movement! send Footage to get surveillance footage or Video to record 5s clip!')
                
             
             
             
            if countr== 400:
                cv2.imwrite("04.jpg", img=frame)
                telegram_bot.sendPhoto(chat_id=562320888, photo=open('/home/sanjay/project/opencv-face/04.jpg', 'rb'), caption='There is movement! send Footage to get surveillance footage or Video to record 5s clip!')
                
            


            if countr== 540:
                cv2.imwrite("04.jpg", img=frame)
                telegram_bot.sendPhoto(chat_id=562320888, photo=open('/home/sanjay/project/opencv-face/04.jpg', 'rb'), caption='There is movement! send Footage to get surveillance footage or Video to record 5s clip!')
                countr=0
            
           
                
                





















	





	cv2.imshow("Frame", frame)







	key = cv2.waitKey(1) & 0xFF

	
	if key == ord("q"):

            break






fps.stop()

print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
result.release()
footage.release()
telegram_bot.sendDocument(chat_id=562320888, document=open('/home/sanjay/project/opencv-face/footage.avi', 'rb'), caption='Surveillance Ended! Here is the footage.')
telegram_bot.sendDocument(chat_id=495615272, document=open('/home/sanjay/project/opencv-face/footage.avi', 'rb'), caption='Surveillance Ended! Here is the footage.')
telegram_bot.sendDocument(chat_id=374931237, document=open('/home/sanjay/project/opencv-face/footage.avi', 'rb'), caption='Surveillance Ended! Here is the footage.')
telegram_bot.sendDocument(chat_id=649640936, document=open('/home/sanjay/project/opencv-face/footage.avi', 'rb'), caption='Surveillance Ended! Here is the footage.')




cv2.destroyAllWindows()
vs.stop()
