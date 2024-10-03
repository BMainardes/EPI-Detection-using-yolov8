import cv2
from ultralytics import YOLO

# Carregar o modelo YOLOv8 treinado, certifique-se de utilizar o diretório de seu modelo
model = YOLO(r"C:\Users\model.pt")

# Inicializar a captura da câmera
# Para que o comando camera = cv2.VideoCapture(0) utilize um sistema de câmeras de segurança, como câmeras IP ou câmeras conectadas à rede, 
# você precisará ajustar o parâmetro de captura para a URL da câmera, ao invés de usar o valor 0, que indica a webcam local. Normalmente, 
# as câmeras de segurança IP fornecem um stream de vídeo acessível via protocolo RTSP ou HTTP.

camera = cv2.VideoCapture(0)  # Use 0 para a webcam padrão ou substitua pelo ID da sua câmera

while True:
    # Capturar o frame da câmera
    ret, frame = camera.read()

    if not ret:
        print("Falha ao capturar imagem da câmera")
        break

    # Fazer o reconhecimento de objetos no frame
    results = model(frame)

    # Desenhar as caixas delimitadoras e rótulos nos objetos detectados
    # Existem outras formar de desenhar as caixas delimitadoras mas esta pareceu a mais agradável
    annotated_frame = results[0].plot()

    # Exibir o frame com os objetos detectados
    cv2.imshow("Reconhecimento de Equipamentos de Segurança", annotated_frame)

    # Sair com a tecla 'q'
    # Você pode utilizar a biblioteca pySerial para se comunicar com o Arduino via USB (porta serial). O código lerá o sinal do Arduino e substituirá o comportamento de cv2.waitKey().
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar todas as janelas
camera.release()
cv2.destroyAllWindows()
