import cv2
import os
env_folder = 'tmp/dump/exp1/episodes/1/4'
image_folder = env_folder
video_name = env_folder+'video.avi'

images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
print(len(images))
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 15, (width,height))
i=0
for image in images:
    print (i)
    i= i+1
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()