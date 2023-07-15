import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import cv2

api_url = "http://localhost:9900/v1/estimatepose"
image_path = os.path.join("brainypi-ai-api-examples", "sample_inputs", "images", "pose2.jpg")

image = cv2.imread(image_path)
retval,image_file = cv2.imencode('.jpg', image)
if retval:
    encoded_image = image_file.tobytes()
    print("encoded")
else:
    print("Image encoding failed.")

response = requests.post(api_url, encoded_image)

if response.status_code == 200:
    response_data = response.json()

    print(response_data)
    pose_points = response_data["result"]["poses"][0]["points"]

    x = [point["x"] for point in pose_points]
    y = [point["y"] for point in pose_points]

    fig, ax = plt.subplots()

    ax.imshow(image)

    ax.scatter(x, y, color='r')

    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)

    circle = patches.Circle(((min_x+max_x)/2, (min_y+max_y)/2), ((max_x - min_x)/2), linewidth=2, edgecolor='r', facecolor='none')

    ax.add_patch(circle)

    outline_points = np.array([[point["x"], point["y"]] for point in pose_points])
    outline_patch = patches.Polygon(outline_points, closed=True, linewidth=2, edgecolor='b', facecolor='none')

    ax.add_patch(outline_patch)
    plt.show()


else:
    print("Error: Response received from the server {}".format(response.status_code))
print("Code execution completed.")
