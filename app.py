from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Carga de modelos
model1 = tf.keras.models.load_model('CNN-Flowers-Model1.h5')
model2 = tf.keras.models.load_model('CNN-Flowers-Model2.h5')

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_with_benchmark(model, image):
    start = time.time()
    predictions = model.predict(image)
    end = time.time()
    time_elapsed = end - start
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = predictions[0]
    return predicted_class, confidence, time_elapsed

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = preprocess_image(filepath)

        # Modelo 1
        prediction1, confidences1, time1 = predict_with_benchmark(model1, image)
        accuracy1 = 0.70

        # Modelo 2
        prediction2, confidences2, time2 = predict_with_benchmark(model2, image)
        accuracy2 = 0.80

        # Prepara porcentajes
        percentages1 = [(class_names[i], round(float(confidences1[i]) * 100, 2)) for i in range(len(class_names))]
        percentages2 = [(class_names[i], round(float(confidences2[i]) * 100, 2)) for i in range(len(class_names))]
        
        # Obtener confianza de la clase predicha
        confidence1_value = next(percent for class_name, percent in percentages1 if class_name == prediction1)
        confidence2_value = next(percent for class_name, percent in percentages2 if class_name == prediction2)

        return render_template("index.html",
                               prediction1=prediction1,
                               prediction2=prediction2,
                               percentages1=percentages1,
                               percentages2=percentages2,
                               class_names=class_names,
                               accuracy1=round(accuracy1 * 100, 2),
                               accuracy2=round(accuracy2 * 100, 2),
                               time1=round(time1, 3),
                               time2=round(time2, 3),
                               filename=filename,
                               confidence1=confidence1_value,
                               confidence2=confidence2_value)

    return render_template("index.html")

if __name__ == '__main__':
    # Obtener puerto de las variables de entorno (Render lo proporciona)
    port = int(os.environ.get('PORT', 5000))
    # Importante: host='0.0.0.0' permite acceso externo
    app.run(host='0.0.0.0', port=port, debug=False)