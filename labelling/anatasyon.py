import os  # Dosya ve klasör işlemleri için kullanılır
import torch  # PyTorch, derin öğrenme kütüphanesi (kareye alınan resmin ne olduğunu tahmin etmek için.)
from pathlib import Path  # Dosya ve klasör yollarını işlemek için
from PIL import Image  # Görüntü dosyalarını açmak ve kaydetmek için

# YOLO modelini yükleme
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  


def detect_and_label_images(input_folder, output_folder):#Fonksiyon oluşturdum ve input ve output klasörleri için bu işlemi yapmasını istedim.
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)  
    # pathlib kütüphanesi ile çıktı klasörünü oluşturdum.

    # 'input_folder' içinde bulunan her bir dosya için işlemleri gerçekleştir
    for image_file in os.listdir(input_folder):
        # Dosya yolunu oluştur
        image_path = os.path.join(input_folder, image_file)
        
        # Nesne tespiti yap
        results = model(image_path)
        # torch kütüphanesi yolo'nun tespit ettiği kare içindeki nesneyi tahmin ediyor.

        
        results.show()
        # 'results' değişkeninin içindeki tahmin sonuçlarını ekranda gösterir.

        # Etiketleme işlemi 
        label = input(f"Resimdeki nesneler için etiket girin: ")
        
        img = Image.open(image_path)
        # PIL kütüphanesi ile resmi açtım.

        
        img.save(os.path.join(output_folder, f"labeled_{image_file}"))  
        # PIL kütüphanesi ile resmi labeled_.jpg olarak kaydetmesini istedim.

        # Etiketleri ve koordinatları  içinde '.txt' dosyası olarak kaydettim.
        with open(os.path.join(output_folder, f"{Path(image_file).stem}.txt"), 'w') as f:
            

            # Her resim için etiket, güven skoru ve koordinatları yazdırdım.
            for *box, conf, cls in results.xyxy[0]:
                # YOLO modelinden elde edilen tahminlerin koordinatlarını ve güven skorlarını işler.
                
                label_str = f"{label} {conf:.2f} "  # Etiket ve güven skorunu yazma

                box_str = ' '.join(map(str, box))  # Koordinatları yazma
                
                f.write(f"{label_str}{box_str}\n")  # Her tahmini bir satıra yazma

# Anatasyon işlemi için resim klasörü ve etiketleme işleminin kaydedileceği klasör yollarını iki adet değişkene atadım.
input_folder = r"C:\Users\Ali\Downloads\archive (18)\cat_dog\cats"  # Resimlerin bulunduğu klasör
output_folder = r"C:\Users\Ali\Downloads\archive (18)\cat_dog\labelling"  # Etiketlenen resimlerin kaydedileceği klasör

detect_and_label_images(input_folder, output_folder) #Oluşturduğum fonksiyonu çağırdım.