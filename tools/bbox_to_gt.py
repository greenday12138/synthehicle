import sys, os
import json
import numpy as np

data_dir = "/home/ubuntu2004/Git/synthehicle/carla/generate_carla_data/scenes_non_overlap/Town05_Opt/day_2024-01-05_13-27-23"

if __name__ is "__main__":
    for cam in os.listdir(data_dir):
        cam_path = os.path.join(data_dir, cam)
        if (not os.path.exists(os.path.join(cam_path, 'gt'))):
            os.makedirs(os.path.join(cam_path, 'gt'))
        print(cam_path)
        frames = os.listdir(os.path.join(cam_path, 'out_bbox'))
        frames.sort()

        frame_bboxs = np.zeros((1, 10), dtype=np.int32)
        for i, frame in enumerate(frames):
            frame_bbox = open(os.path.join(cam_path, 'out_bbox', frame), 'r', encoding='utf-8')
            js = frame_bbox.read()
            data = json.loads(js)
            bboxs = np.zeros((len(data["vehicle_id"]), 10), dtype=np.int32)
            for j in range(len(data["vehicle_id"])):
                bboxs[j] = np.array([i+1, data["vehicle_id"][j], 
                                     int(data["bboxes"][j][0][0]), 
                                     int(data["bboxes"][j][0][1]), 
                                     int(data["bboxes"][j][1][0] - data["bboxes"][j][0][0]), 
                                     int(data["bboxes"][j][1][1] - data["bboxes"][j][0][1]),
                                     1, -1, -1, -1])
            np.sort(bboxs, axis=1)
            # print(bboxs)
            frame_bboxs = np.concatenate((frame_bboxs, bboxs), axis = 0)
        frame_bboxs = np.delete(frame_bboxs, 0, 0)
        np.savetxt(os.path.join(cam_path, 'gt', 'gt.txt'), frame_bboxs, fmt='%d', delimiter= ',')
