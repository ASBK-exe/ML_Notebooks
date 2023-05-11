from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import os

model = tf.keras.models.load_model('/Users/ademsalehbk/Documents/GitHub/ML_Notebooks/Tomato Disease Detection Project/models/v.2.0')

class_names = ['Tomato with Bacterial spot',
               'Tomato with Early blight',
               'Tomato with Late blight',
               'Tomato with Leaf Mold',
               'Tomato with Septoria leaf spot',
               'Tomato with Two spotted spider mite',
               'Tomato with Target Spot',
               'Tomato with Yellow Leaf Curl Virus',
               'Tomato with mosaic virus',
               'Healthy Tomato']

app = Flask(__name__)

# Define the upload folder for images
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static', 'uploads')

# Create the upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        image = Image.open(file)
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        label = class_names[np.argmax(prediction)]
        accuracy = round(np.max(prediction) * 100, 2)
        
        
        return render_template('predict.html', label=label, accuracy=accuracy, image_file=file.filename)
    else:
        return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
