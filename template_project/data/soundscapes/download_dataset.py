from __future__ import unicode_literals
import os
import youtube_dl
import pandas as pd
from subprocess import call

def get_subdirectory(row):
    if row['anthropophony'] == 1:
        return 'raw-soundscape-dataset/anthropophony/road_traffic'
    if row['biophony'] == 1:
        return 'raw-soundscape-dataset/biophony/birds'
    if row['biophony'] == 2:
        return 'raw-soundscape-dataset/biophony/frogs'
    if row['geophony'] == 1:
        return 'raw-soundscape-dataset/geophony/wind'
    if row['geophony'] == 2:
        return 'raw-soundscape-dataset/geophony/rain'

def download_audio(url, output_file):
    print("Download {} into {}".format(url, output_file))
    #call(['youtube-dl', url, '--extract-audio',
    #    '--audio-format=wav', '--output={}'.format(output_file)])
    call(['youtube-dl', '-f', 'bestaudio', '--extract-audio', '--audio-format',
        'wav', url, '-o',
        output_file])

if __name__ == '__main__':
    
    data = pd.read_csv('data.csv')

#    for i, row in data.iterrows():
#        print(row)
#        print(row['youtube-url'])
#        sub_directory = get_subdirectory(row)
#        audio_id = row['youtube-url'].split('?v=')[1]
#        output_file = '{}.wav'.format(audio_id)
#        print(output_file)
#        download_audio(row['youtube-url'], os.path.join(sub_directory, output_file))

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192'
        }],
        'postprocessor_args': [
            '-ar', '22050'
        ],
        'prefer_ffmpeg': True,
        'keepvideo': False
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(data['youtube-url'])
