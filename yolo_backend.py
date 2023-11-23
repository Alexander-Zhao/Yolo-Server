import hashlib

import cv2
import matplotlib
import torch
from skimage import io

# Define YOLO model ops
yolo_model = torch.hub.load(
    "ultralytics/yolov5", "yolov5s"
)  # or yolov5n - yolov5x6, custom
cmap = matplotlib.cm.get_cmap("jet")


def hash_to_range(number, N):
    # Convert the number to a string and then encode it to bytes
    byte_representation = str(number).encode("utf-8")

    # Use SHA-256 hash function
    hashed = hashlib.sha256(byte_representation)

    # Convert the hash to an integer
    hash_integer = int(hashed.hexdigest(), 16)

    # Map the hash to the range [1, N]
    return 1 + (hash_integer % N)


def predict(img_path):
    results = yolo_model(img_path)
    return results.pandas().xyxy[0], results.names


def predict_and_draw(img_path, out_img_path):
    img = io.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result, class_names = predict(img_path)
    result["color"] = result.apply(
        lambda x: tuple(
            c * 255
            for c in cmap(
                hash_to_range(x["class"], len(class_names)) / len(class_names)
            )
        )[:3],
        axis=1,
    )

    for _, el in result.iterrows():
        p1, p2 = (el["xmin"], el["ymin"]), (el["xmax"], el["ymax"])
        p1, p2 = tuple(map(int, p1)), tuple(map(int, p2))
        cv2.rectangle(img, p1, p2, el["color"], 2)

    cv2.imwrite(str(out_img_path), img)

    result_dict = result.to_dict(orient="index")
    return {
        "boxes": [v for _, v in result_dict.items()],
    }

def object_count (data)
    #data = {
    #    'boxes': [
            # List of dictionaries containing object information 
            # (detected by yolo server, and collected from result_dict)
     #   ],
    #}

    # Dictionary to map categories (vehicle types)
    category_mapping = {
        'car': 'vehicle',
        'bus': 'vehicle',
        'truck': 'vehicle',
        'person': 'person'
        # we can add more mappings if needed for other categories
    }

    # Dictionary to store counts
    category_counts = {'vehicle': 0, 'person': 0}

    # Iterate through each box and count specified objects
    for box in data['boxes']:
        object_name = box['name']
    
        # Check if the object name is in the mapping
        if object_name in category_mapping:
            # Map the object to its category
            category = category_mapping[object_name]
        
            # Increment the count for the category
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1
                
    # Display the counts for vehicles and persons
    for category, count in category_counts.items():
        print(f"{category.capitalize()} count: {count}")
        
    return (category_counts)
