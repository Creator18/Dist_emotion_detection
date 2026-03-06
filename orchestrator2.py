from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)

# List of emotion nodes' URLs
emotion_nodes = {
    'anger': 'http://localhost:8001/predict',
    'joy': 'http://localhost:8002/predict',
    'sadness': 'http://localhost:8003/predict',
    'surprise': 'http://localhost:8004/predict',
    'fear': 'http://localhost:8005/predict'
}

# Confidence threshold to consider a secondary emotion
CONFIDENCE_THRESHOLD = 0.1

# Function to send request to the emotion node and get the response
def get_prediction(emotion, text):
    try:
        response = requests.post(emotion_nodes[emotion], json={'text': text})
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error in {emotion} node: {response.text}")
    except Exception as e:
        print(f"Error in {emotion} node: {e}")
        return {"prediction": "Not Emotion", "confidence": 0.0}

# Function to determine the primary and secondary emotions
def determine_emotions(predictions):
    # Get the primary emotion (highest confidence)
    primary_emotion = max(predictions, key=lambda x: x['confidence'])

    # Get secondary emotions that are close in confidence to the primary emotion
    secondary_emotions = {
        emotion['emotion']: round(emotion['confidence'] * 100, 2)  # Convert confidence to percentage
        for emotion in predictions
        if abs(emotion['confidence'] - primary_emotion['confidence']) <= CONFIDENCE_THRESHOLD
           and emotion != primary_emotion
    }

    return primary_emotion, secondary_emotions


# Route for the main interface
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['sentence']
    predictions = []

    # Get predictions from all emotion nodes
    for emotion in emotion_nodes.keys():
        prediction = get_prediction(emotion, text)
        predictions.append({
            'emotion': emotion,
            'confidence': prediction['confidence']
        })

    # Determine primary and secondary emotions
    primary_emotion, secondary_emotions = determine_emotions(predictions)

    # Render output page with results
    return render_template(
        'output.html',
        primary_emotion=primary_emotion['emotion'].capitalize(),
        secondary_emotions=secondary_emotions
    )


if __name__ == '__main__':
    app.run(debug=True)
