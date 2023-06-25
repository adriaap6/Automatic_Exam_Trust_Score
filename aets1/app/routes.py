from app import app
from flask import render_template


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route("/classification", methods = ['GET', 'POST'])
def classification():
	return render_template("classifications.html")

@app.route('/submit', methods=['POST'])
def predict_sound():
    if 'file' not in request.files:
        resp = jsonify({'message': 'No sound in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('file')
    filename = "temp_sound.wav"
    errors = {}
    success = False
    for file in files:
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
        else:
            errors["message"] = 'File type of {} is not allowed'.format(file.filename)

    if not success:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp
    sound_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Load sound file and extract features
    data, sr = sf.read(sound_path)
    
    data, sr = sf.read(sound_path)
    # Perform feature extraction on the sound file (e.g., using librosa)
    # Replace the following code with your feature extraction code
    # features = extract_features(data, sr)
    features = np.random.rand(128)  # Dummy features for illustration

    # Reshape features for prediction
    features = np.expand_dims(features, axis=0)

    # Predict
    prediction_array_cnn = modelcnn.predict(features)

    # Prepare API response
    class_names = ['With Pest', 'Without Pest']

    return render_template("classifications.html", predictioncnn=class_names[np.argmax(prediction_array_cnn)],
                           confidencecnn='{:.2f}%'.format(100 * np.max(prediction_array_cnn)))


if __name__ == '__main__':
    # app.debug = True
    app.run(debug = True)
    