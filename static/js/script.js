// ===== VARIABLES GLOBALES =====
const themeSwitch = document.getElementById('themeSwitch');
const themeIcons = document.querySelectorAll('.theme-icon');
const html = document.documentElement;
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('image-preview');
const loader = document.getElementById('loader');
const form = document.getElementById('uploadForm');
const fileInfo = document.getElementById('fileInfo');
const submitBtn = document.getElementById('submitBtn');

const allowedFormats = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff'];

// ===== FUNCIONES UTILITARIAS =====
function getFileExtension(filename) {
  return filename.split('.').pop().toLowerCase();
}

function isValidFile(file) {
  const extension = getFileExtension(file.name);
  return allowedFormats.includes(extension);
}

function isWebPFile(file) {
  return getFileExtension(file.name) === 'webp';
}

function showFileError(message) {
  dropzone.classList.add('error');
  fileInfo.innerHTML = `<span style="color: var(--error);"> ${message}</span>`;
  submitBtn.disabled = true;
  setTimeout(() => {
    dropzone.classList.remove('error');
  }, 3000);
}

function showFileSuccess(filename) {
  dropzone.classList.remove('error');
  fileInfo.innerHTML = `<span style="color: var(--success);"> Archivo seleccionado: ${filename}</span>`;
  submitBtn.disabled = false;
}

// ===== TEMA OSCURO/CLARO =====
function initializeTheme() {
  // Comprobar preferencia del sistema
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const savedTheme = localStorage.getItem('theme');
  
  // Aplicar tema guardado o preferencia del sistema
  if (savedTheme) {
    html.setAttribute('data-theme', savedTheme);
    updateThemeIcons(savedTheme);
  } else if (prefersDark) {
    html.setAttribute('data-theme', 'dark');
    updateThemeIcons('dark');
  }
}

function updateThemeIcons(theme) {
  themeIcons.forEach(icon => {
    if (icon.getAttribute('data-theme') === theme) {
      icon.classList.add('active');
    } else {
      icon.classList.remove('active');
    }
  });
}

function setupThemeSwitch() {
  themeSwitch.addEventListener('click', (e) => {
    if (e.target.classList.contains('theme-icon')) {
      const theme = e.target.getAttribute('data-theme');
      html.setAttribute('data-theme', theme);
      localStorage.setItem('theme', theme);
      updateThemeIcons(theme);
    }
  });
}

// ===== MANEJO DE ARCHIVOS =====
function handleFile(file) {
  if (!file) return;

  if (isWebPFile(file)) {
    showFileError('Los archivos WebP no son compatibles. Usa JPG o PNG.');
    fileInput.value = '';
    imagePreview.style.display = 'none';
    return;
  }

  if (!isValidFile(file)) {
    showFileError('Formato no permitido. Usa: JPG, PNG, GIF, BMP o TIFF.');
    fileInput.value = '';
    imagePreview.style.display = 'none';
    return;
  }

  showFileSuccess(file.name);
  
  const reader = new FileReader();
  reader.onload = function(e) {
    imagePreview.src = e.target.result;
    imagePreview.style.display = 'block';
  }
  reader.readAsDataURL(file);
}

// ===== DRAG & DROP =====
function setupFileUpload() {
  fileInput.addEventListener('change', function(e) {
    handleFile(e.target.files[0]);
  });

  dropzone.addEventListener('click', () => fileInput.click());
  
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
  
  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });
  
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    fileInput.files = e.dataTransfer.files;
    handleFile(file);
  });

  form.addEventListener('submit', function(e) {
    const file = fileInput.files[0];
    if (!file) {
      e.preventDefault();
      showFileError('Por favor selecciona un archivo');
      return;
    }
    
    if (isWebPFile(file) || !isValidFile(file)) {
      e.preventDefault();
      return;
    }
    
    loader.style.display = 'block';
    submitBtn.disabled = true;
    submitBtn.textContent = 'Procesando...';
  });
}

// ===== BARRAS DE CONFIANZA =====
function setupConfidenceBars() {
  document.querySelectorAll('.confidence-bar').forEach(bar => {
    const width = bar.getAttribute('data-width');
    if (width) {
      // Aplicar después de un pequeño retraso para permitir la animación
      setTimeout(() => {
        bar.style.width = `${width}%`;
      }, 100);
    }
  });
}

// ===== MENSAJES FLASH =====
function setupFlashMessages() {
  setTimeout(() => {
    const flashMessages = document.querySelectorAll('.flash-message');
    flashMessages.forEach(msg => {
      msg.style.transition = 'opacity 0.5s ease-out';
      msg.style.opacity = '0';
      setTimeout(() => msg.remove(), 500);
    });
  }, 5000);
}

// ===== GRÁFICAS =====
function setupCharts() {
  const chartDataElement = document.getElementById('chart-data');
  if (!chartDataElement) return;

  const chartData = JSON.parse(chartDataElement.textContent);
  
  const labels = chartData.labels;
  const model1Data = chartData.model1Data;
  const model2Data = chartData.model2Data;
  
  // Configuración común para ambas gráficas
  const chartConfig = {
    type: 'bar',
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        title: {
          display: true,
          text: 'Distribución de probabilidades',
          font: {
            size: 16,
            weight: '600'
          },
          color: 'var(--text)',
          padding: {
            top: 10,
            bottom: 20
          }
        },
        tooltip: {
          backgroundColor: 'rgba(30, 30, 30, 0.85)',
          padding: 12,
          cornerRadius: 8,
          displayColors: false,
          callbacks: {
            label: function(context) {
              return `${context.dataset.label}: ${context.raw}%`;
            },
            title: function(context) {
              return labels[context[0].dataIndex];
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          grid: {
            color: 'var(--border)'
          },
          ticks: {
            callback: function(value) {
              return value + '%';
            },
            font: {
              size: 12
            },
            color: 'var(--text-light)'
          },
          title: {
            display: true,
            text: 'Probabilidad',
            font: {
              size: 14,
              weight: '600'
            },
            color: 'var(--text)',
            padding: {
              top: 10,
              bottom: 10
            }
          }
        },
        x: {
          grid: {
            display: false
          },
          ticks: {
            maxRotation: 45,
            minRotation: 30,
            font: {
              size: 12
            },
            color: 'var(--text)'
          },
          title: {
            display: true,
            text: 'Clases de flores',
            font: {
              size: 14,
              weight: '600'
            },
            color: 'var(--text)',
            padding: {
              top: 15
            }
          }
        }
      },
      layout: {
        padding: {
          left: 10,
          right: 10,
          top: 10,
          bottom: 30
        }
      },
      animation: {
        duration: 1000,
        easing: 'easeOutQuart'
      }
    }
  };
  
  // Gráfica para Modelo 1
  const ctx1 = document.getElementById('chart1');
  if (ctx1) {
    new Chart(ctx1.getContext('2d'), {
      ...chartConfig,
      data: {
        labels: labels,
        datasets: [{
          label: 'Confianza (%)',
          data: model1Data,
          backgroundColor: [
            '#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe', '#e0e7ff'
          ],
          borderColor: '#4f46e5',
          borderWidth: 1,
          borderRadius: 8,
          borderSkipped: false,
        }]
      }
    });
  }
  
  // Gráfica para Modelo 2
  const ctx2 = document.getElementById('chart2');
  if (ctx2) {
    new Chart(ctx2.getContext('2d'), {
      ...chartConfig,
      data: {
        labels: labels,
        datasets: [{
          label: 'Confianza (%)',
          data: model2Data,
          backgroundColor: [
            '#10b981', '#34d399', '#6ee7b7', '#a7f3d0', '#d1fae5'
          ],
          borderColor: '#059669',
          borderWidth: 1,
          borderRadius: 8,
          borderSkipped: false,
        }]
      }
    });
  }
}

// ===== INICIALIZACIÓN =====
document.addEventListener('DOMContentLoaded', function() {
  initializeTheme();
  setupThemeSwitch();
  setupFileUpload();
  setupConfidenceBars();
  setupFlashMessages();
  setupCharts();
});