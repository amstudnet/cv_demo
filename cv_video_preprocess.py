import os
import json
import cv2

import shutil

def video_to_frames(video_path,size=None):
    """
    to do ??
    """
    cap = cv2.VideoCapture(video_path)

    frames = []

    while 1:
        ret,frame = cap.read()#ret 代表是否讀到圖片 frame代表擷取到一偵圖片
        if ret:
            if size:
                frame = cv2.resize(frame,size)
            frames.append(frame)
        else:
            break
    cap.release()

    return frames

def convert_frames_to_video(frame_array, path_out, size, fps=25):
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def extract_frame_as_video(src_video_path, start_frame, end_frame):
    frames = video_to_frames(src_video_path)
    return frames[start_frame: end_frame+1]


def extract_all_yt_instance(x):
    cnt = 1
    if not os.path.exists("videos"):
        os.mkdir("videos")
    for entry in x :
        instances = entry["instances"]

        for inst in instances:
            url = inst["url"]
            video_id = inst["video_id"]
            if 'youtube' in url or 'youtu.be' in url:
                cnt+=1
                yt_identifier = url[-11:]

                src_video_path = os.path.join('raw_videos', yt_identifier + '.mp4')
                dst_video_path = os.path.join('videos', video_id + '.mp4')

                if not os.path.exists(src_video_path):continue
                if os.path.exists(dst_video_path):
                    print("{}exists.".format(dst_video_path))
                    continue
                start_frame = inst['frame_start'] - 1 
                end_frame = inst['frame_end'] - 1
                if end_frame <=0:
                    shutil.copyfile(src_video_path,dst_video_path)
                    continue
                selected_frames = extract_frame_as_video(src_video_path,start_frame,end_frame)
                # to look frame size
                size = selected_frames[0].shape[:2][::-1]

                convert_frames_to_video(selected_frames,dst_video_path,size)

                print(cnt,dst_video_path)

def main():

    x = json.load(open('WLASL_v0.3.json'))
    extract_all_yt_instance(x)

if __name__ =="__main__":
    main()