# MoodSync

MoodSync is a web application that captures your mood through an uploaded photo and generates a personalized music playlist based on the predicted emotion. This project consists of a React front end, a Flask back end, a PyTorch model for emotion prediction, and an AWS Lambda function for processing music playlists.

## How it works

1. **Front End (React)**
    - Users can upload a photo through the web interface.
    - The uploaded photo is sent to the Flask backend for processing.

2. **Back End (Flask)**
    - Flask server receives the uploaded image.
    - The image is processed using a PyTorch model to predict the user's emotion.
    - The result is sent to an AWS Lambda function for further processing.

3. **PyTorch Model**
    - The PyTorch model analyzes the image and predicts the user's emotion.
    - Emotion classes: 'anger', 'focus', 'happy', 'tired'.

4. **AWS Lambda Function**
    - Receives the emotion prediction result from the Flask server.
    - Retrieves music playlist data from an S3 bucket based on the predicted emotion.
    - Sends the playlist data back to the Flask server.

5. **S3 Bucket**
    - Periodically updates playlist files.
    - Ensures that the Lambda function receives the latest version of the playlist.

6. **Back End (Flask) - Part 2**
    - Flask server receives the music playlist data from the Lambda function.
    - Displays featured music tailored to the user on the web interface.
