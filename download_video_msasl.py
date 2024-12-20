import os
import json
import time
import sys
import urllib.request
from multiprocessing.dummy import Pool  # 目前未使用，未來可以用來實現並行下載

import random

import logging
# 設定日誌，會將日誌紀錄到名為 download_<timestamp>_msasl.log 的文件中，同時也會輸出到控制台
logging.basicConfig(filename='download_{}_msasl.log'.format(int(time.time())), filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# 設定使用的下載工具，可以選擇 "yt-dlp" 或 "youtube-dl"
youtube_downloader = "yt-dlp"

def request_video(url, referer=''):
    """
    發送 HTTP 請求以下載影片，並返回下載的數據。
    """
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    
    headers = {'User-Agent': user_agent}
    
    if referer:
        headers['Referer'] = referer  # 如果有提供 Referer 頭部，則加入

    request = urllib.request.Request(url, None, headers)  # 創建帶有自定義標頭的請求

    logging.info('Requesting {}'.format(url))  # 記錄請求的 URL
    response = urllib.request.urlopen(request)  # 發送請求
    data = response.read()  # 讀取返回的數據

    return data  # 返回影片的二進位數據

def save_video(data, saveto):
    """
    將下載的影片數據儲存到指定的檔案中。
    """
    with open(saveto, 'wb+') as f:
        f.write(data)  # 寫入檔案

    # 請稍作停頓，以免對伺服器造成過大負擔
    time.sleep(random.uniform(0.5, 1.5))  # 隨機延遲 0.5 到 1.5 秒

def download_youtube(url, dirname, video_id):
    """
    處理 YouTube 影片下載，這裡留空未實現，因為使用 yt-dlp 處理 YouTube。
    """
    raise NotImplementedError("Urllib cannot deal with YouTube links.")
def check_youtube_dl_version():
    """
    檢查 youtube-dl 或 yt-dlp 的版本，確保其可用。
    """
    ver = os.popen(f'{youtube_downloader} --version').read()

    # 如果無法找到 yt-dlp，則報錯
    assert ver, f"{youtube_downloader} cannot be found in PATH. Please verify your installation."
def download_yt_videos(indexfile, saveto='test_videos'):
    """
    下載 YouTube 影片，使用 yt-dlp 或 youtube-dl 進行下載。
    """
    content = json.load(open(indexfile))  # 讀取 JSON 文件
    
    if not os.path.exists(saveto):
        os.mkdir(saveto)  # 如果儲存資料夾不存在，則創建
    
    for entry in content:
        gloss = entry['text']  # 影片的手語詞條名稱
        video_url = entry['url']
        video_id = entry['label']
        if 'youtube' not in video_url and 'youtu.be' not in video_url:
            continue  # 如果不是 YouTube 影片，則跳過
        if os.path.exists(os.path.join(saveto, video_url[-11:] + '.mp4')) or os.path.exists(os.path.join(saveto, video_url[-11:] + '.mkv')):
            logging.info('YouTube videos {} already exists.'.format(video_url))  # 如果影片已存在，則跳過
            continue
        else:
            cmd = f"{youtube_downloader} \"{{}}\" -o \"{{}}%(id)s.%(ext)s\""  # 設定下載命令
            cmd = cmd.format(video_url, saveto + os.path.sep)
            rv = os.system(cmd)  # 執行命令
            if not rv:
                logging.info('Finish downloading youtube video url {}'.format(video_url))  # 成功下載
            else:
                logging.error('Unsuccessful downloading - youtube video url {}'.format(video_url))  # 下載失敗
            # 隨機延遲，避免過度請求
            time.sleep(random.uniform(1.0, 1.5))


if __name__ == '__main__':
    #logging.info('Start downloading non-youtube videos.')  # 記錄開始下載非 YouTube 影片
    #download_nonyt_videos('MSASL_train.json')  # 下載非 YouTube 影片

    check_youtube_dl_version()  # 檢查 youtube-dl/yt-dlp 工具是否安裝
    logging.info('Start downloading youtube videos.')  # 記錄開始下載 YouTube 影片
    #download_yt_videos('MSASL_train.json')  # 下載 YouTube 影片
    #print("train finish")
    #download_yt_videos('MSASL_val.json')
    #print("val finish")
    download_yt_videos('MSASL_test.json')
    print('test finish')
