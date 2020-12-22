# Smart-Home-Security-using-Telegram-Chatbot
Using the combination of face recognition and motion detection.

// Objective //

An affordable home security system using Telegram chatbot.
Uses Face recognition algorithm along with Motion detection to give correct intrusion alerts
Sends out intrusion alert with picture when unknown person enters frame.
User can request surveillance footage through commands.
The system alerts user when it can’t detect a face but there is movements.

// Problem Statement //

Smart home security is very expensive to set up.
Existing system uses motion detection alone which produces false alerts.
A subscription amount is necessary for storage of surveillance footage.
Our system uses motion detection and face recognition to filter out false results.
Surveillance footage is stored in telegram server free of cost.
Cost effective to set up.

// First Create the dataset using build_face_dataset.py //

// USAGE>>>
python build_face_dataset.py --cascade haarcascade_frontalface_default.xml --output dataset/<dataset-name>

// Next Train a Machine learning model using the Datasets

























































































































































Most of the code is used from https://www.pyimagesearch.com/.
We combined two separate algorithms and added our own to make this system work.





