# Face-recognition-with-dlib
### Face recognition and identification using unique facial landmarks

⧈ This script is a simple ***face recognition program***. It first reads in a series of sample images from the "player_images" directory, and uses the **face_recognition module** to encode these images. When the script is run, it captures video from the default webcam and uses face_recognition to **find and encode any faces** in the video frames. 

⧈ It then compares these encodings to the encodings of the sample images and, if it finds a match, displays the name of the person in the sample image on the screen. The script also updates a file called "Registered_Names.csv" with the name of the person and the current date and time whenever a match is found. 

⧈ Finally, the script will continue running until the user presses the "q" key, at which point it will properly close all windows and release the webcam.
