import os
import glob

src_dir = "./Dataset/UCF-101/" # Dir containing the videos
des_dir = './Dataset/UCF-101/' # Output dir to save the videos

for file in os.listdir(src_dir):
    vid_files = glob.glob1(src_dir+file, '*.avi')
    des_dir = "./Dataset/UCF-101/frames/"
    for vid in vid_files:
        des_dir_path = os.path.join(des_dir+file, vid[:-4])
        if not os.path.exists(des_dir_path):
            os.makedirs(des_dir_path)
        os.system('ffmpeg -i ' + os.path.join(src_dir+file, vid) + ' -qscale:v 2 ' + des_dir_path + '/frames%05d.jpg')