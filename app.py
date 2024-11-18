from flask import Flask, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

def load_my_model(model_path):
    # Load the model from the .h5 file
    model = load_model(model_path)
    return model

model = load_my_model('model_weights.h5')

frame_buffer = []

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (64, 64))
    # Add any additional preprocessing steps here
    return resized_frame

def predict(frames):
    # Process frames and make predictions
    # Replace this with your actual prediction code
    frames = np.array(frames)
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    prediction = model.predict(frames)
    return prediction

def webcam_feed():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        frame = preprocess_frame(frame)

        # Add the frame to the buffer
        frame_buffer.append(frame)
        if len(frame_buffer) > 20:
            frame_buffer.pop(0)  # Remove the oldest frame if buffer size exceeds 20

        if len(frame_buffer) == 20:
            # Perform prediction on the frame buffer
            prediction = predict(frame_buffer)

            # Display prediction on the frame (you can customize this)
            # For example, draw the predicted pose on the frame

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(webcam_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
