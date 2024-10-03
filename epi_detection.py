import cv2
from ultralytics import YOLO

# Carregar o modelo YOLOv8 treinado
model = YOLO(r"C:\Users\Bruno\runs\detect\train10\weights\best.pt")

# Inicializar a captura da câmera
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
    annotated_frame = results[0].plot()

    # Exibir o frame com os objetos detectados
    cv2.imshow("Reconhecimento de Equipamentos de Segurança", annotated_frame)

    # Sair com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar todas as janelas
camera.release()
cv2.destroyAllWindows()
