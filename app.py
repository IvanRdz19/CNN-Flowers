from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from threading import Thread
import logging

# Configurar TensorFlow para optimización
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Crear el directorio de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Variables globales para modelos
model1 = None
model2 = None
models_loaded = False

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def load_models():
    """Carga los modelos de forma optimizada con manejo de errores de compatibilidad"""
    global model1, model2, models_loaded
    try:
        app.logger.info("Cargando modelos...")
        
        # Configurar TensorFlow para compatibilidad
        K = tf.keras.backend
        K.clear_session()
        
        # Intentar cargar con diferentes métodos
        try:
            # Método 1: Carga normal
            model1 = tf.keras.models.load_model('CNN-Flowers-Model1.h5', compile=False)
            model2 = tf.keras.models.load_model('CNN-Flowers-Model2.h5', compile=False)
        except Exception as e1:
            app.logger.warning(f"Método 1 falló: {str(e1)}")
            try:
                # Método 2: Con custom_objects vacío
                model1 = tf.keras.models.load_model('CNN-Flowers-Model1.h5', 
                                                  compile=False, 
                                                  custom_objects={})
                model2 = tf.keras.models.load_model('CNN-Flowers-Model2.h5', 
                                                  compile=False, 
                                                  custom_objects={})
            except Exception as e2:
                app.logger.warning(f"Método 2 falló: {str(e2)}")
                # Método 3: Carga con safe_mode
                model1 = tf.keras.models.load_model('CNN-Flowers-Model1.h5', 
                                                  compile=False,
                                                  safe_mode=False)
                model2 = tf.keras.models.load_model('CNN-Flowers-Model2.h5', 
                                                  compile=False,
                                                  safe_mode=False)
        
        # Verificar que los modelos se cargaron correctamente
        if model1 is None or model2 is None:
            raise Exception("Los modelos no se cargaron correctamente")
        
        # Warm up - hacer una predicción dummy para inicializar
        dummy_input = np.random.random((1, 128, 128, 3)).astype(np.float32)
        _ = model1.predict(dummy_input, verbose=0)
        _ = model2.predict(dummy_input, verbose=0)
        
        models_loaded = True
        app.logger.info("Modelos cargados exitosamente")
        
    except Exception as e:
        app.logger.error(f"Error cargando modelos: {str(e)}")
        app.logger.error(f"Tipo de error: {type(e).__name__}")
        models_loaded = False

def preprocess_image_from_memory(file):
    """Procesa imagen directamente desde memoria sin guardar"""
    try:
        # Leer imagen desde memoria
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        app.logger.error(f"Error procesando imagen: {str(e)}")
        return None

def predict_with_benchmark(model, image):
    """Predicción optimizada"""
    try:
        start = time.time()
        predictions = model.predict(image, verbose=0)  # verbose=0 para menos logs
        end = time.time()
        time_elapsed = end - start
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = predictions[0]
        return predicted_class, confidence, time_elapsed
    except Exception as e:
        app.logger.error(f"Error en predicción: {str(e)}")
        return None, None, 0

@app.route('/health')
def health_check():
    """Health check para Render"""
    return jsonify({"status": "healthy", "models_loaded": models_loaded})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Verificar que los modelos estén cargados
        if not models_loaded:
            return render_template("index.html", 
                                 error="Los modelos aún se están cargando. Intenta de nuevo en unos segundos.")
        
        # Verificar archivo
        if 'file' not in request.files:
            return render_template("index.html", error="No se seleccionó archivo")
            
        file = request.files['file']
        if not file or file.filename == '':
            return render_template("index.html", error="No se seleccionó archivo válido")

        # Verificar tipo de archivo
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return render_template("index.html", 
                                 error="Formato no válido. Use: PNG, JPG, JPEG, GIF, BMP, WEBP")

        try:
            # Procesar imagen desde memoria (más rápido)
            image = preprocess_image_from_memory(file)
            if image is None:
                return render_template("index.html", error="Error procesando la imagen")

            # Predicciones
            result1 = predict_with_benchmark(model1, image)
            result2 = predict_with_benchmark(model2, image)
            
            if result1[0] is None or result2[0] is None:
                return render_template("index.html", error="Error en las predicciones")
            
            prediction1, confidences1, time1 = result1
            prediction2, confidences2, time2 = result2
            
            # Valores fijos de accuracy (puedes cambiarlos)
            accuracy1 = 0.70
            accuracy2 = 0.80

            # Prepara porcentajes
            percentages1 = [(class_names[i], round(float(confidences1[i]) * 100, 2)) 
                          for i in range(len(class_names))]
            percentages2 = [(class_names[i], round(float(confidences2[i]) * 100, 2)) 
                          for i in range(len(class_names))]
            
            # Obtener confianza de la clase predicha
            confidence1_value = next(percent for class_name, percent in percentages1 
                                   if class_name == prediction1)
            confidence2_value = next(percent for class_name, percent in percentages2 
                                   if class_name == prediction2)

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
                                 filename=secure_filename(file.filename),
                                 confidence1=confidence1_value,
                                 confidence2=confidence2_value)
        
        except Exception as e:
            app.logger.error(f"Error general: {str(e)}")
            return render_template("index.html", 
                                 error="Error procesando la solicitud. Intenta de nuevo.")

    return render_template("index.html")

# Cargar modelos al importar el módulo
Thread(target=load_models).start()

if __name__ == '__main__':
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Cargar modelos antes de iniciar
    load_models()
    
    # Obtener puerto de las variables de entorno
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)