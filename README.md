# Smart_selfie

 I came across this advertisement for Oppo — the phone automatically captures a selfie when the beautiful actress smiles at the camera. This seemed like a pretty easy challenge given the wonderful dlib library from Python.




This project is about smart selfie using python programming. It detects a smile on the face and accordingly takes a selfie-and saves it into your drive. The full procedure is been described step by step.

Libraries used in this code are :

    OpenCV
    Numpy
    DLIB ( Toolkit containing Machine Learning algorithms )


The following pipeline would be followed : 

    Recognize the face in the video
    Recognize facial landmarks
    Calculate the smile parameter
    If sp > threshold, Take a selfie
