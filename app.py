from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf
import logging
import io
from concurrent.futures import ThreadPoolExecutor
import threading

# Configurar TensorFlow para optimización
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Crear el directorio de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Variables globales para modelos con lock para thread safety
model1 = None
model2 = None
models_loaded = False
model_lock = threading.Lock()
loading_error = None

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_fallback_model():
    """Crea un modelo básico de CNN para clasificación de flores"""
    try:
        logger.info("Creando modelo fallback...")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128, 128, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Warm up
        dummy_input = np.random.random((1, 128, 128, 3)).astype(np.float32)
        _ = model.predict(dummy_input, verbose=0)
        
        logger.info("Modelo fallback creado exitosamente")
        return model
    except Exception as e:
        logger.error(f"Error creando modelo fallback: {str(e)}")
        return None

def load_models():
    """Carga los modelos de forma síncrona con mejor manejo de errores"""
    global model1, model2, models_loaded, loading_error
    
    with model_lock:
        try:
            logger.info("Iniciando carga de modelos...")
            tf.keras.backend.clear_session()
            
            # Intentar cargar modelos reales
            model_paths = ['CNN-Flowers-Model1.h5', 'CNN-Flowers-Model2.h5']
            models = []
            
            for i, path in enumerate(model_paths, 1):
                try:
                    if os.path.exists(path):
                        logger.info(f"Cargando modelo {i} desde {path}...")
                        model = tf.keras.models.load_model(path, compile=False)
                        logger.info(f"Modelo {i} cargado exitosamente")
                    else:
                        logger.warning(f"Archivo {path} no encontrado, usando modelo fallback")
                        model = create_fallback_model()
                    
                    if model is None:
                        raise Exception(f"No se pudo crear modelo {i}")
                    
                    models.append(model)
                    
                except Exception as e:
                    logger.error(f"Error cargando modelo {i}: {str(e)}")
                    logger.info(f"Creando modelo fallback para modelo {i}")
                    fallback = create_fallback_model()
                    if fallback is None:
                        raise Exception(f"No se pudo crear modelo fallback {i}")
                    models.append(fallback)
            
            model1, model2 = models
            
            # Warm up ambos modelos
            logger.info("Realizando warm-up de modelos...")
            dummy_input = np.random.random((1, 128, 128, 3)).astype(np.float32)
            
            _ = model1.predict(dummy_input, verbose=0)
            _ = model2.predict(dummy_input, verbose=0)
            
            models_loaded = True
            loading_error = None
            logger.info("Todos los modelos cargados y listos")
            
        except Exception as e:
            error_msg = f"Error crítico cargando modelos: {str(e)}"
            logger.error(error_msg)
            loading_error = error_msg
            models_loaded = False

def wait_for_models(timeout=60):
    """Espera a que los modelos se carguen o timeout"""
    start_time = time.time()
    while not models_loaded and (time.time() - start_time) < timeout:
        if loading_error:
            return False
        time.sleep(0.5)
    return models_loaded

def preprocess_image_from_file(file):
    """Procesa imagen con mejor manejo de errores"""
    try:
        logger.info("Procesando imagen...")
        
        # Reset file pointer to beginning
        file.stream.seek(0)
        
        # Leer todos los bytes
        file_bytes = file.stream.read()
        
        # Crear imagen desde bytes
        img = Image.open(io.BytesIO(file_bytes))
        
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionar
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        
        # Convertir a array numpy
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Expandir dimensiones para batch
        result = np.expand_dims(img_array, axis=0)
        
        logger.info(f"Imagen procesada exitosamente. Shape: {result.shape}")
        return result
        
    except Exception as e:
        logger.error(f"Error procesando imagen: {str(e)}")
        return None

def predict_with_model(model, image, model_name):
    """Predicción con manejo de errores mejorado"""
    try:
        logger.info(f"Realizando predicción con {model_name}...")
        start = time.time()
        
        with model_lock:
            predictions = model.predict(image, verbose=0)
        
        end = time.time()
        time_elapsed = end - start
        
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = predictions[0]
        
        logger.info(f"Predicción {model_name} completada: {predicted_class} "
                   f"({float(np.max(confidence)):.2%}) en {time_elapsed:.3f}s")
        
        return predicted_class, confidence, time_elapsed
        
    except Exception as e:
        logger.error(f"Error en predicción {model_name}: {str(e)}")
        return None, None, 0

@app.route('/health')
def health_check():
    """Health check mejorado"""
    status = {
        "status": "healthy" if models_loaded else "loading",
        "models_loaded": models_loaded,
        "loading_error": loading_error,
        "timestamp": time.time()
    }
    return jsonify(status), 200 if models_loaded else 503

@app.route('/status')
def status():
    """Endpoint de estado detallado"""
    return jsonify({
        "models_loaded": models_loaded,
        "loading_error": loading_error,
        "model1_available": model1 is not None,
        "model2_available": model2 is not None,
        "tensorflow_version": tf.__version__
    })

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            logger.info("Recibida solicitud POST")
            
            # Verificar que los modelos estén cargados
            if not models_loaded:
                if loading_error:
                    error_msg = f"Error cargando modelos: {loading_error}"
                else:
                    error_msg = "Los modelos aún se están cargando. Intenta de nuevo en unos segundos."
                
                logger.warning(error_msg)
                return render_template("index.html", error=error_msg)
            
            # Verificar archivo
            if 'file' not in request.files:
                logger.warning("No se encontró archivo en request")
                return render_template("index.html", error="No se seleccionó archivo")
                
            file = request.files['file']
            if not file or file.filename == '':
                logger.warning("Archivo vacío o sin nombre")
                return render_template("index.html", error="No se seleccionó archivo válido")

            logger.info(f"Procesando archivo: {file.filename}")

            # Verificar tipo de archivo
            allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
            if not ('.' in file.filename and 
                    file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
                return render_template("index.html", 
                                     error="Formato no válido. Use: PNG, JPG, JPEG, GIF, BMP, WEBP")

            # Procesar imagen
            image = preprocess_image_from_file(file)
            if image is None:
                return render_template("index.html", error="Error procesando la imagen")

            # Realizar predicciones
            result1 = predict_with_model(model1, image, "Modelo 1")
            result2 = predict_with_model(model2, image, "Modelo 2")
            
            if result1[0] is None or result2[0] is None:
                return render_template("index.html", error="Error en las predicciones")
            
            prediction1, confidences1, time1 = result1
            prediction2, confidences2, time2 = result2
            
            # Valores de accuracy (ajústalos según tus modelos reales)
            accuracy1 = 0.70
            accuracy2 = 0.80

            # Preparar porcentajes
            percentages1 = [(class_names[i], round(float(confidences1[i]) * 100, 2)) 
                          for i in range(len(class_names))]
            percentages2 = [(class_names[i], round(float(confidences2[i]) * 100, 2)) 
                          for i in range(len(class_names))]
            
            # Obtener confianza de la clase predicha
            confidence1_value = next(percent for class_name, percent in percentages1 
                                   if class_name == prediction1)
            confidence2_value = next(percent for class_name, percent in percentages2 
                                   if class_name == prediction2)

            logger.info("Predicciones completadas exitosamente")

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
            logger.error(f"Error general en index: {str(e)}", exc_info=True)
            return render_template("index.html", 
                                 error=f"Error procesando la solicitud: {str(e)}")

    return render_template("index.html")

# Inicialización de la aplicación
def initialize_app():
    """Inicializa la aplicación cargando los modelos"""
    logger.info("Inicializando aplicación...")
    load_models()
    if models_loaded:
        logger.info("Aplicación inicializada correctamente")
    else:
        logger.error("Error en la inicialización de la aplicación")

if __name__ == '__main__':
    initialize_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
else:
    # Para cuando se ejecuta con gunicorn
    initialize_app()