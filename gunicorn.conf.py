import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"
backlog = 2048

# Worker processes - Para ML es mejor usar pocos workers
workers = 1
worker_class = "sync"
worker_connections = 1000

# Timeouts más largos para ML
timeout = 300  # 5 minutos para carga de modelos y predicciones
keepalive = 5
graceful_timeout = 30

# Memory management
max_requests = 50  # Reiniciar workers más frecuentemente para ML
max_requests_jitter = 10

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'cnn-flowers-app'

# Server mechanics
preload_app = True  # Importante para cargar modelos una vez
daemon = False
pidfile = None
user = None
group = None

# Límites de memoria (importante para ML)
limit_request_line = 8190
limit_request_fields = 200
limit_request_field_size = 8190

# Worker timeout específico para startup
worker_timeout = 120

# SSL
keyfile = None
certfile = None

# Configuraciones específicas para ML
def when_ready(server):
    """Callback cuando el servidor está listo"""
    server.log.info("Servidor listo - Modelos de ML inicializados")

def worker_int(worker):
    """Callback para manejar interrupciones del worker"""
    worker.log.info("Worker interrupted")

def pre_fork(server, worker):
    """Callback antes de hacer fork del worker"""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    """Callback después de hacer fork del worker"""
    server.log.info("Worker spawned (pid: %s)", worker.pid)