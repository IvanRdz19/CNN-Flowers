import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"
backlog = 2048

# Worker processes
workers = 1  # Solo 1 worker para evitar cargar modelos múltiples veces
worker_class = "sync"
worker_connections = 1000
timeout = 120  # Timeout más largo para ML
keepalive = 2

# Restart workers after this many requests, to prevent memory leaks
max_requests = 100
max_requests_jitter = 50

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = 'cnn-flowers-app'

# Server mechanics
preload_app = True  # Importante: precarga la app para cargar modelos una vez
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# SSL
keyfile = None
certfile = None