#!/usr/bin/env python

import os

#print os.system("sleep 2m &")
os.system('./htbin/processvideo.sh ./videos/video_input.mp4 ./videos/video_output.mp4')
print "\n\n\n\nYa Termino\n\n\n\n"