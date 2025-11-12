📸 Cámara Interactiva con Gestos (OpenCV y MediaPipe)

Este proyecto es una aplicación de cámara web en tiempo real, escrita en Python, que te permite controlar una multitud de funciones usando únicamente gestos con las manos. Utiliza OpenCV para todo el procesamiento de imágenes y MediaPipe para la detección de gestos.


🚀 Características Principales

👋 Control por Gestos: Utiliza la detección de manos de MediaPipe para una interacción sin contacto.

🎥 Grabación de Vídeo: Inicia y detiene grabaciones con un simple gesto.

📷 Captura de Fotos: Toma fotos con una cuenta atrás activada por gestos.

🔍 Zoom Digital: Acerca y aleja la imagen con un gesto intuitivo de dos manos.

☀️ Ajuste de Brillo/Contraste: Controla dinámicamente el brillo y el contraste de la imagen en tiempo real.

🎨 Filtros en Tiempo Real: Cambia entre 6 filtros diferentes, incluyendo:

Normal

Escala de Grises

Detección de Bordes (Canny)

Desenfoque (Blur) con Detección de Caras

Segmentación de Color (Verde HSV)

Efecto Cómic (Adaptive Threshold)

👤 Detección de Caras: Activa y desactiva la detección de caras (Viola-Jones) para ver los cuadros delimitadores.

🖐️ Visualización de Puntos: Muestra y oculta el esqueleto de la mano de MediaPipe para depuración.

⚙️ Instalación y Ejecución

Sigue estos pasos para poner en marcha el proyecto en tu máquina local.

Prerrequisitos

Python 3.7+

Una cámara web conectada

Pasos

Clona el repositorio:

git clone [https://github.com/AlbertooCh/CamaraInteractiva_OpenCV.git](https://github.com/AlbertooCh/CamaraInteractiva_OpenCV.git)
cd CamaraInteractiva_OpenCV


Crea un entorno virtual (recomendado):

python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate


Instala las dependencias:
No se necesita un archivo requirements.txt grande. Solo instala las bibliotecas principales:

pip install opencv-python mediapipe numpy


Ejecuta el script:
(Asegúrate de que tu cámara web no esté siendo utilizada por otra aplicación).

python CamaraInteractiva_PuntosMano.py
