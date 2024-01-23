import cv2
import os


def img2video(cam_list):
    cams = os.listdir(cam_list)

    for cam in cams: 

        image_folder = f'{cam_list}/{cam}/out_rgb'
        video_name = f'{cam_list}/{cam}/video.mp4'

        images = [img for img in os.listdir(
            image_folder) if img.endswith(".jpg")]
        images.sort()
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        video = cv2.VideoWriter(video_name, fourcc, 20, (width, height), True)

        for i,image in enumerate(images):
            # if i > 6600:
            #     break
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()


# img2video("/home/ubuntu2004/Git/synthehicle/carla/generate_carla_data/scenes_non_overlap/Town05_Opt/day_2024-01-14_18-45-07")
# def main():


# if __name__ == '__main__':
#     main()
