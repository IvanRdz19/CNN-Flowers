from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.secret_key = 'clasificador_flores_cnn_2024'  # Necesario para usar flash messages
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Extensiones permitidas
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Asegúrate que la carpeta de uploads exista
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Carga de modelos
model1 = tf.keras.models.load_model('CNN-Flowers-Model1.h5')
model2 = tf.keras.models.load_model('CNN-Flowers-Model2.h5')

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def allowed_file(filename):
    """Verifica si el archivo tiene una extensión permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_webp_file(filename):
    """Verifica si el archivo es WebP"""
    return filename.lower().endswith('.webp')

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
        if 'file' not in request.files:
            flash('No se seleccionó ningún archivo', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No se seleccionó ningún archivo', 'error')
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            
            # Verificar si es un archivo WebP
            if is_webp_file(filename):
                flash('Error: Los archivos WebP no son compatibles. Por favor, usa formatos como JPG o PNG', 'error')
                return redirect(request.url)
            
            # Verificar si el archivo tiene una extensión permitida
            if not allowed_file(filename):
                flash('Error: Formato de archivo no permitido. Usa, JPG, PNG', 'error')
                return redirect(request.url)
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                
                # Intentar procesar la imagen
                image = preprocess_image(filepath)

                # Predicción con modelo 1
                prediction1, confidences1, time1 = predict_with_benchmark(model1, image)
                accuracy1 = 0.70  # Si quieres puedes calcular o cambiar este valor

                # Predicción con modelo 2
                prediction2, confidences2, time2 = predict_with_benchmark(model2, image)
                accuracy2 = 0.80  # Igual que arriba

                # Porcentajes para mostrar en la interfaz
                percentages1 = [(class_names[i], round(float(confidences1[i]) * 100, 2)) for i in range(len(class_names))]
                percentages2 = [(class_names[i], round(float(confidences2[i]) * 100, 2)) for i in range(len(class_names))]

                # Obtener confianza de la clase predicha
                confidence1_value = next(percent for class_name, percent in percentages1 if class_name == prediction1)
                confidence2_value = next(percent for class_name, percent in percentages2 if class_name == prediction2)

                flash('Imagen procesada correctamente', 'success')
                
                return render_template("index.html",
                                       prediction1=prediction1,
                                       prediction2=prediction2,
                                       percentages1=percentages1,
                                       percentages2=percentages2,
                                       class_names=class_names,  # ¡AGREGAR ESTA LÍNEA!
                                       accuracy1=round(accuracy1 * 100, 2),
                                       accuracy2=round(accuracy2 * 100, 2),
                                       time1=round(time1, 3),
                                       time2=round(time2, 3),
                                       filename=filename,
                                       confidence1=confidence1_value,
                                       confidence2=confidence2_value)
            
            except Exception as e:
                flash(f'Error al procesar la imagen: {str(e)}', 'error')
                # Eliminar archivo si hay error
                if os.path.exists(filepath):
                    os.remove(filepath)
                return redirect(request.url)

    return render_template("index.html", 
                         prediction1=None, 
                         prediction2=None,
                         class_names=class_names)

if __name__ == '__main__':
    app.run(debug=True)