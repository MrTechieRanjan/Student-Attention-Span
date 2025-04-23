from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("C:/Users/Ayush/Desktop/yolov3_project/runs/detect/train11/weights/best.pt")

    metrics = model.val(data="C:/Users/Ayush/Desktop/yolov3_project/dataset.yaml")

    # Extracting overall metrics
    print("Precision:", metrics.box.p.mean())  # Mean Precision
    print("Recall:", metrics.box.r.mean())  # Mean Recall
    print("mAP@50:", metrics.box.map50)  # Mean Average Precision at 50% IoU
    print("mAP@50-95:", metrics.box.map)  # Mean Average Precision at IoU 50-95%

    # Print per-class metrics
    for i, cls in enumerate(metrics.box.ap_class_index):
        print(f"Class {metrics.names[cls]} - Precision: {metrics.box.p[i]:.3f}, Recall: {metrics.box.r[i]:.3f}, AP@50: {metrics.box.ap[i]:.3f}, AP@50-95: {metrics.maps[i]:.3f}")
