from ultralytics import YOLO

# A função principal do script. Isso garante que o código dentro do bloco só será executado
# se o arquivo for rodado diretamente e não importado como um módulo em outro código.
if __name__ == '__main__':
    
    # Carregar o modelo YOLOv8
    # Substitua este caminho pelo seu modelo personalizado, caso tenha feito algum ajuste ou queira usar outro tipo.
       model = YOLO('yolov8n.pt')  # ou o caminho para o seu modelo YOLO

    # Comando usado para treinar o modelo, seguidos dos parâmetros para a execução do mesmo
    model.train(
        # Caminho para o arquivo data.yaml, que contém a configuração do dataset. 
        # Esse arquivo define as classes, caminhos dos conjuntos de treinamento, validação, etc.
        # Certifique-se de que o caminho está correto e que o dataset foi preparado no formato adequado para YOLOv8.
        data=r'C:\Users\Bruno\Downloads\EPI detection.v12i.yolov8\data.yaml',
        
        # Define o número de épocas (passagens completas pelo dataset) para o treinamento.
        # Quanto mais épocas, maior a chance do modelo aprender bem o padrão, mas o treinamento também demora mais.
        # Para datasets grandes, 100 épocas costuma ser um bom ponto de partida, mas você pode ajustar conforme necessário.
        epochs=100,
        
        # Define o tamanho do batch (quantidade de imagens processadas por vez durante o treinamento).
        # Aqui, 8 é um valor razoável, mas depende da quantidade de memória disponível na sua GPU.
        # Você pode ajustar o tamanho do batch se encontrar problemas de memória.
        batch=8,
        
        # Define o tamanho das imagens para o treinamento. '640' é uma resolução comum para YOLOv8,
        # que equilibra bem a precisão e a velocidade de treinamento. Valores maiores (ex: 1280) podem melhorar a precisão,
        # mas exigem mais memória e tempo de processamento.
        imgsz=640,
        
        # Especifica o dispositivo de treinamento. O valor 'cuda' indica que o treinamento será feito em uma GPU,
        # o que acelera significativamente o processo em comparação ao uso de uma CPU ('cpu').
        # Se não tiver uma GPU CUDA disponível, substitua por 'cpu' (mas o treinamento será muito mais lento).
        # Alguns erros de compatilibilidade com o CUDA  pode acontecer em relação ao PyTorch e o torchvision, certifique-se de utiliza-los corretamente
        device='cuda'
    )
