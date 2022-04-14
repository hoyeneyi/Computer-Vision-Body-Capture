import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf
import logging

W = 848
H = 480

# Configure depth and color streams...
# ...from Camera 1
pipeline_1 = rs.pipeline()
config_1 = rs.config()

config_1.enable_device('146222252203')
config_1.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config_1.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
# ...from Camera 2
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device('146222253424')
config_2.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config_2.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# Start streaming from both cameras
print("[INFO] start streaming...")
pipeline_1.start(config_1)
pipeline_2.start(config_2)

aligned_stream = rs.align(rs.stream.color)  # alignment between color and depth
point_cloud = rs.pointcloud()

print("[INFO] loading model...")
PATH_TO_CKPT = r"frozen_inference_graph.pb"
# download model from: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API#run-network-in-opencv

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
# code source of tensorflow model loading: https://www.geeksforgeeks.org/ml-training-image-classifier-using-tensorflow-object-detection-api/

num = 0
try:
    while True:
        height1 = 0
        height2 = 0
        # Camera 1
        # Wait for a coherent pair of frames: depth and color
        frames_1 = pipeline_1.wait_for_frames()
        depth_frame_1 = frames_1.get_depth_frame()
        color_frame_1 = frames_1.get_color_frame()
        points = point_cloud.calculate(depth_frame_1)
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)  # xyz

        if not depth_frame_1 or not color_frame_1:
            continue
        # Convert images to numpy arrays
        depth_image_1 = np.asanyarray(depth_frame_1.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())
        scaled_size = (int(W), int(H))
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap_1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_1, alpha=0.5), cv2.COLORMAP_JET)

        image_expanded = np.expand_dims(color_image_1, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                 feed_dict={image_tensor: image_expanded})

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)

        # print("[INFO] drawing bounding box on detected objects...")
        # print("[INFO] each detected object has a unique color")

        for idx in range(int(num)):
            class_ = classes[idx]
            score = scores[idx]
            box = boxes[idx]
            # print(" [DEBUG] class : ", class_, "idx : ", idx, "num : ", num)

            if score > 0.65 and class_ == 1:  # 1 for human
                left = box[1] * W
                top = box[0] * H
                right = box[3] * W
                bottom = box[2] * H

                width = right - left
                height = bottom - top
                bbox = (int(left), int(top), int(width), int(height))
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                # draw box
                cv2.rectangle(color_image_1, p1, p2, (255, 0, 0), 2, 1)

                # x,y,z of bounding box
                obj_points = verts[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])].reshape(-1,
                                                                                                                     3)
                zs = obj_points[:, 2]

                z = np.median(zs)

                ys = obj_points[:, 1]
                ys = np.delete(ys, np.where(
                    (zs < z - 1) | (zs > z + 1)))  # take only y for close z to prevent including background

                my = np.amin(ys, initial=1)
                My = np.amax(ys, initial=-1)

                height1 = (My - my)  # add next to rectangle print of height using cv library
                height1 = float("{:.2f}".format(height1))
                print("Camera1 height is: ", height1, "[m]")
                height1_txt = str(height1) + "[m]"

                # Write some Text
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (p1[0], p1[1] + 20)
                fontScale = 1
                fontColor = (255, 255, 255)
                lineType = 2
                cv2.putText(color_image_1, height1_txt,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)

        # Camera 2
        # Wait for a coherent pair of frames: depth and color
        frames_2 = pipeline_2.wait_for_frames()
        depth_frame_2 = frames_2.get_depth_frame()
        color_frame_2 = frames_2.get_color_frame()
        points = point_cloud.calculate(depth_frame_2)
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)  # xyz

        if not depth_frame_2 or not color_frame_2:
            continue
        # Convert images to numpy arrays
        depth_image_2 = np.asanyarray(depth_frame_2.get_data())
        color_image_2 = np.asanyarray(color_frame_2.get_data())
        scaled_size = (int(W), int(H))
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap_2 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_1, alpha=0.5), cv2.COLORMAP_JET)

        image_expanded = np.expand_dims(color_image_2, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                 feed_dict={image_tensor: image_expanded})

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)

        # print("[INFO] drawing bounding box on detected objects...")
        # print("[INFO] each detected object has a unique color")

        for idx in range(int(num)):
            class_ = classes[idx]
            score = scores[idx]
            box = boxes[idx]
            # print(" [DEBUG] class : ", class_, "idx : ", idx, "num : ", num)

            if score > 0.65 and class_ == 1:  # 1 for human
                left = box[1] * W
                top = box[0] * H
                right = box[3] * W
                bottom = box[2] * H

                width = right - left
                height = bottom - top
                bbox = (int(left), int(top), int(width), int(height))
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                # draw box
                cv2.rectangle(color_image_2, p1, p2, (255, 0, 0), 2, 1)

                # x,y,z of bounding box
                obj_points = verts[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])].reshape(-1,
                                                                                                                     3)
                zs = obj_points[:, 2]

                z = np.median(zs)

                ys = obj_points[:, 1]
                ys = np.delete(ys, np.where(
                    (zs < z - 1) | (zs > z + 1)))  # take only y for close z to prevent including background

                my = np.amin(ys, initial=1)
                My = np.amax(ys, initial=-1)

                height2 = (My - my)  # add next to rectangle print of height using cv library
                height2 = float("{:.2f}".format(height2))
                print("Camera2 height is: ", height2, "[m]")
                height2_txt = str(height2) + "[m]"

                # Write some Text
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (p1[0], p1[1] + 20)
                fontScale = 1
                fontColor = (255, 255, 255)
                lineType = 2
                cv2.putText(color_image_2, height2_txt,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)

        avg_height = (height1 + height2) / 2
        print("Average: " + str(avg_height))
        # Stack all images horizontally
        images = np.hstack((color_image_1, color_image_2))

        # Show images from both cameras

        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite(r'C:\Users\JBCMSI\Desktop\Project\CameraD\LeftImage\color' + str(num) + '.png', color_image_1)
            # cv2.imwrite(r'C:\Users\JBCMSI\Desktop\Project\CameraD\LeftImage\depth' + str(num)+'.png',depth_colormap_1)
            cv2.imwrite(r'C:\Users\JBCMSI\Desktop\Project\CameraD\RightImage\color' + str(num) + '.png', color_image_2)
            # cv2.imwrite(r'C:\Users\JBCMSI\Desktop\Project\CameraD\RightImage\depth' + str(num)+'.png',depth_colormap_2)
            print("Save")
            num += 1
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', images)


finally:

    # Stop streaming
    pipeline_1.stop()
    pipeline_2.stop()
