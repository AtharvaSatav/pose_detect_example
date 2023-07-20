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

    pose_points = response_data["result"]["poses"][0]["points"]

    x = [point["x"] for point in pose_points]
    y = [point["y"] for point in pose_points]

    fig, ax = plt.subplots()

    ax.imshow(image)

    for index, (x_coord, y_coord) in enumerate(zip(x, y),):
       ax.scatter(x_coord, y_coord, color='r')
       # ax.text(x_coord, y_coord, str(index), color='g')


    connections = [
    (0, 1, 'r-'),
    (1, 2, 'r-'),
    (2, 4, 'r-'),
    (4, 5, 'r-'),
    (5, 8, 'r-'),
    (5, 9, 'r-'),
    (8, 10, 'r-'),
    (5, 11, 'r-'),
    (11, 14, 'r-'),
    (14, 16, 'r-'),
    (11, 13, 'r-'),
    (13, 15, 'r-')
    ]

    for start_index, end_index, style in connections:
       start_point = pose_points[start_index]
       end_point = pose_points[end_index]
       ax.plot([start_point['x'], end_point['x']], [start_point['y'], end_point['y']], style, linewidth=3.0)

    plt.show()


else:
    print("Error: Response received from the server {}".format(response.status_code))
print("Code execution completed.")
