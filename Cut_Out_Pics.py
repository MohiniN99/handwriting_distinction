import os
import cv2
import json
import shutil
from sklearn.model_selection import train_test_split

with open('coco_train.json', 'r') as f:
    data = json.load(f)


image_folder = '/Users/mohini/Library/Mobile Documents/com~apple~CloudDocs/ML_Projekt_Handschrift/Mohini_Vorlage/neueTestdatenMohini.png'
output_folder = 'test'


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

images = {img['id']: os.path.join(image_folder, img['file_name']) for img in data['images']}


for annotation in data['annotations']:
    image_id = annotation['image_id']
    image_path = images[image_id]

    
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Image {image_path} not found or unable to load.")
        continue

    x, y, w, h = annotation['bbox']
    
    # Breite und Höhe auf 200
    new_w = 200
    new_h = 200
    
    # neue Koordinaten für die Mitte der Box
    center_x = x + w // 2
    center_y = y + h // 2
    
    # neue obere linke Ecke
    new_x = max(0, center_x - new_w // 2)
    new_y = max(0, center_y - new_h // 2)
    
    # Begrenzung der Werte
    if new_x + new_w > image.shape[1]:
        new_x = image.shape[1] - new_w
    if new_y + new_h > image.shape[0]:
        new_y = image.shape[0] - new_h
    
    
    crop_image = image[new_y:new_y + new_h, new_x:new_x + new_w]
    
    category_name = next((cat['name'] for cat in data['categories'] if cat['id'] == annotation['category_id']), 'unknown')
    category_folder = os.path.join(output_folder, category_name)
    if not os.path.exists(category_folder):
        os.makedirs(category_folder)
    
  
    output_filename = f'{annotation["id"]}.png'
    output_path = os.path.join(category_folder, output_filename)
    cv2.imwrite(output_path, crop_image)
    print(f'Saved: {output_path}')

# Aufteilen der Daten in Training, Validierung und Test
def split_data(input_folder, output_folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    categories = os.listdir(input_folder)
    
    for category in categories:
        category_path = os.path.join(input_folder, category)
        images = os.listdir(category_path)
        
        train, test = train_test_split(images, test_size=test_ratio, random_state=42)
        train, val = train_test_split(train, test_size=val_ratio/(train_ratio+val_ratio), random_state=42)
        
        for split, split_name in [(train, 'train'), (val, 'val'), (test, 'test')]:
            split_folder = os.path.join(output_folder, split_name, category)
            if not os.path.exists(split_folder):
                os.makedirs(split_folder)
            for img in split:
                src_path = os.path.join(category_path, img)
                dst_path = os.path.join(split_folder, img)
                shutil.copyfile(src_path, dst_path)


split_data(output_folder, output_folder)

print("Daten erfolgreich in Trainings-, Validierungs- und Testsets aufgeteilt.")
