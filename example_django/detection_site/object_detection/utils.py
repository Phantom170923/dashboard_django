import cv2
import numpy as np
from django.core.files.base import ContentFile
from .models import ImageFeed, DetectedObject
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
import io

VOC_LABELS = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]




def process_image(image_feed_id, model_type=None):
    try:
        image_feed = ImageFeed.objects.get(id=image_feed_id)
        image_path = image_feed.image.path

        if model_type == 'model_1':
            model_path = 'object_detection/mobilenet_iter_73000.caffemodel'
            config_path = 'object_detection/mobilenet_ssd_deploy.prototxt'
            net = cv2.dnn.readNetFromCaffe(config_path, model_path)

            img = cv2.imread(image_path)
            if img is None:
                print("Failed to load image")
                return False

            h, w = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)

            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.6:
                    class_id = int(detections[0, 0, i, 1])
                    class_label = VOC_LABELS[class_id]
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    label = f"{class_label}: {confidence:.2f}"
                    cv2.putText(img, label, (startX+5, startY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    DetectedObject.objects.create(
                        image_feed=image_feed,
                        object_type=class_label,
                        location=f"{startX},{startY},{endX},{endY}",
                        confidence=float(confidence)
                    )

            result, encoded_img = cv2.imencode('.jpg', img)
            if result:
                content = ContentFile(encoded_img.tobytes(), f'processed_{image_feed.image.name}')
                image_feed.processed_image.save(content.name, content, save=True)

        if model_type == 'model_2':

            im = Image.open(image_path)
            im_resize = im

            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

            inputs = processor(images=im_resize, return_tensors="pt")
            outputs = model(**inputs)

            # convert outputs (bounding boxes and class logits) to COCO API
            # let's only keep detections with score > 0.9
            target_sizes = torch.tensor([im_resize.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            box2 = []
            label2 = []
            score2 = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                print(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )
                box2.append(box)
                label2.append(model.config.id2label[label.item()])
                score2.append(float(score))

                # for i in box2:
                draw = ImageDraw.Draw(im_resize)
                draw.rectangle((box), outline='red', width=2)


                DetectedObject.objects.create(
                    image_feed=image_feed,
                    object_type=model.config.id2label[label.item()],
                    location=f"{box}",
                    confidence=float(score)
                )

            buf = io.BytesIO()
            im_resize.save(buf, format='JPEG')
            byte_im = buf.getvalue()
            content = ContentFile(byte_im, f'processed_{image_feed.image.name}')
            image_feed.processed_image.save(content.name, content, save=True)

        return True

    except ImageFeed.DoesNotExist:
        print("ImageFeed not found.")
        return False
