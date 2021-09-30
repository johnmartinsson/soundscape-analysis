import glob
import os
from subprocess import call
# download data and put into directories
# TODO
source_dir = 'raw-soundscape-dataset'
output_dir = 'soundscape-dataset'
sub_dirs = ['anthropophony', 'geophony', 'biophony']

for sub_dir in sub_dirs:
    print(sub_dir)
    wav_dir = os.path.join(source_dir, sub_dir)
    wav_files = glob.glob(os.path.join(wav_dir, '*.wav'))
    for wav_file in wav_files:
        print(wav_file)
        basename = os.path.basename(wav_file)
        if not os.path.exists(os.path.join(output_dir, sub_dir)):
            os.makedirs(os.path.join(output_dir, sub_dir))

        # resample to 22050
        call(['sox', os.path.join(source_dir, sub_dir, basename), '-r', '22050',
            os.path.join(output_dir, sub_dir, '22050_' + basename)])
        # mix to mono
        call(['sox', os.path.join(output_dir, sub_dir, '22050_' + basename),
            os.path.join(output_dir, sub_dir, '22050_mono_' + basename), 'remix', '1,2'])
        # split into 2 second segments
        #call(['sox', os.path.join(output_dir, sub_dir, '22050_mono_' + basename),
        #    os.path.join(output_dir, sub_dir, basename), 'trim 0 15 : newfile : restart'])
        call(['ffmpeg', '-i', os.path.join(output_dir, sub_dir, '22050_mono_' + basename), '-f',
            'segment', '-segment_time', '2', '-c', 'copy',
            os.path.join(output_dir, sub_dir,
                '{}_%05d.wav'.format(basename.split('.')[0]))])
        # clean-up
        call(['rm', os.path.join(output_dir, sub_dir, '22050_' + basename)])
        call(['rm', os.path.join(output_dir, sub_dir, '22050_mono_' + basename)])
