from ultralytics import YOLO
import os
import shutil

def predict_multiple_images():
    # Step 1: Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'best.pt')
    input_dir = os.path.join(base_dir, 'input_image')
    output_dir = os.path.join(base_dir, 'output_image')

    # Step 2: Clear old output folder
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Step 3: Load model
    model = YOLO(model_path)

    # Step 4: Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("❌ No image files found in input_image folder.")
        return
    input_image_paths = [os.path.join(input_dir, f) for f in image_files]

    # Step 5: Run predictions & save directly to output folder
    results = model.predict(
        source=input_image_paths,
        save=True,
        imgsz=640,
        project=output_dir,  # direct save location
        name=''              # no subfolder
    )

    # Step 6: Check detections for each image
    for i, r in enumerate(results):
        num_detections = len(r.boxes)
        print(f"🖼 Image: {image_files[i]} → Detections found: {num_detections}")
        if num_detections > 0:
            print(r.boxes)  # detailed detection info

    print("✅ All predictions saved directly in:", output_dir)

if __name__ == "__main__":
    predict_multiple_images()
