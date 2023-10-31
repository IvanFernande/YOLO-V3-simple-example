"""
Created on Mon Oct 30 16:23:43 2023

@author: ivanf
"""

import cv2 as cv
import numpy as np
import time

def load_yolo():
    net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    return net

def load_classes():
    with open('coco.names', 'r') as file:
        classes = file.read().strip().split('\n')
    return classes

def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detect_objects(net, classes, image, confidence_threshold=0.5, nms_threshold=0.4):
    h, w = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(get_output_layers(net))

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x, center_y, width, height = (detection[:4] * np.array([w, h, w, h])).astype(int)
                x, y = int(center_x - width / 2), int(center_y - height / 2)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            objects.append({
                'class': classes[class_ids[i]],
                'confidence': confidences[i],
                'box': boxes[i]
            })

    return objects

def main():
    try:
        image = cv.imread('yolo_prueba.png')

        net = load_yolo()
        classes = load_classes()

        confidence_threshold = 0.5  # Establece el umbral de confianza
        objects = detect_objects(net, classes, image, confidence_threshold)
        output_image = image.copy()

        for obj in objects:
            x, y, w, h = obj['box']
            color = np.random.randint(0, 255, size=3).tolist()
            cv.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
            text = f'{obj["class"]} ({obj["confidence"]:.2f})'
            cv.putText(output_image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Guardar la imagen de salida
        cv.imwrite('output_image.jpg', output_image)

    except Exception:
        print("An error occurred:")


main()
