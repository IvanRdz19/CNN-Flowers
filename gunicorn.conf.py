import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"
backlog = 1024

# Workers optimizados para velocidad
workers = 1  # 1 worker para evitar overhead de carga de modelos
worker_class = "sync"
worker_connections = 500

# Timeouts optimizados
timeout = 60  # Reducido para forzar respuestas más rápidas
keepalive = 2
graceful_timeout = 15

# Memory management agresivo
max_requests = 200  # Más requests antes de reiniciar
max_requests_jitter = 20

# Logging mínimo para velocidad
accesslog = None  # Desactivar access log para velocidad
errorlog = "-"
loglevel = "warning"  # Solo warnings y errors

# Process naming
proc_name = 'fast-cnn-flowers'

# Server mechanics optimizados
preload_app = True
daemon = False
reuse_port = True  # Mejora performance en Linux

# Límites optimizados
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# Buffer sizes optimizados
worker_tmp_dir = "/tmp"  # Usar RAM disk si está disponible

# Configuraciones para CPU
worker_class = "sync"  # Mejor para CPU-bound tasks como ML

# Callbacks optimizados
def when_ready(server):
    server.log.info("🚀 Servidor optimizado listo")

def worker_int(worker):
    worker.log.info("Worker interrumpido limpiamente")

# Variables de entorno adicionales para optimización
import os
os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'
os.environ['MALLOC_MMAP_THRESHOLD_'] = '131072'