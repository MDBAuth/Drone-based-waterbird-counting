import os
import json
from PIL import Image
from pathlib import Path
from tqdm import tqdm

class CocoDataset:
    def __init__(self, train_data_dir):
        self.train_data_dir = Path(train_data_dir)

    def add_existing_annotations(self, annotations):
        i=1
        j=1
        info = {'description': 'COCO-formatted dataset'}
        licenses = ''
        categories = []
        category_dict = {}
        image_metadata = []
        detections = []
        for filepath, objs in tqdm(annotations.items(), desc="Adding existing annotations"):
            img = Image.open(filepath)
            filename = os.path.basename(filepath)
            img.save(self.train_data_dir / filename)
            width, height = img.size

            image_metadata.append({
                'id': i,
                'file_name': filename,
                'height': height,
                'width': width,
                'license': None,
                'coco_url': None
            })
            for obj in objs:
                name = obj.get('label', None)
                supercategory = obj.get('supercategory', None)
                if name not in [category['name'] for category in categories]:
                    id = len(categories)
                    categories.append(
                        {
                            'id': id,
                            'name': name,
                            'supercategory': supercategory
                        })
                    category_dict[name] = id
                x,y,w,h = obj.get('bbox')
                detections.append({
                    'id': j,
                    'image_id': i,
                    'category_id': category_dict[name],
                    'bbox': [x*width, y*height, w*width, h*height],
                    'area': [w*width*h*height]
                })
                j+=1
            i+=1
        
        self.dataset = {
            'info': info,
            'licenses': licenses,
            'categories': categories,
            'images': image_metadata,
            'annotations': detections
        }

    def add_negative_samples(self, filepaths):
        i = len(self.dataset['images'])
        for filepath in filepaths:
            img = Image.open(filepath)
            filename = os.path.basename(filepath)
            img.save(self.train_data_dir / filename)
            width, height = img.size

            self.dataset['images'].append({
                'id': i,
                'file_name': filename,
                'height': height,
                'width': width,
                'license': None,
                'coco_url': None
            })
            i+=1

    def export_dataset(self, train_coco):
        with open(train_coco, 'w') as fw:
            json.dump(self.dataset, fw)