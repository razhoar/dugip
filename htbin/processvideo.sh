#!/bin/bash

# "/home/w1ndman/Dropbox/estadisticas_futboleras/partidos/Benjamines Rommel Fdez A 0-7 Adeje A 1ª Parte-m9VOsmaRgBU.mp4"
# "/tmp/toronja/Benjamines.mp4"
# ./processvideo.sh "/home/w1ndman/Desktop/bejamines_cut.mp4" "/home/w1ndman/Benjamines_processed.mp4"

INPUT_FILE=$1
OUPUT_FILE=$2

echo "Processing:"
echo "Input:" $INPUT_FILE
echo "Output:" $OUPUT_FILE

#cp "$INPUT_FILE" "$OUPUT_FILE"

SPLITTEDINPUTDIR=/tmp/vp/splitted_input/
SPLITTEDPROCESSEDDIR=/tmp/vp/splitted_processed/
rm -Rf "$SPLITTEDINPUTDIR"
rm -Rf "$SPLITTEDPROCESSEDDIR"

mkdir -p $SPLITTEDINPUTDIR
mkdir -p $SPLITTEDPROCESSEDDIR

rm -f ./videos/video_output.mp4

# Split frames
#python splitframes.py
#/usr/bin/mplayer -ss 00:01:11 -endpos 6 $INPUT_FILE -vo jpeg:outdir="$SPLITTEDINPUTDIR"
echo "SPLITTING"
mplayer $INPUT_FILE -vo jpeg:outdir="$SPLITTEDINPUTDIR" > /dev/null 2>&1 

# Processvideo
python ./htbin/blob_detection.py $SPLITTEDINPUTDIR $SPLITTEDPROCESSEDDIR

# JOIN
echo ffmpeg -r 25 -qscale 2 -i "$SPLITTEDPROCESSEDDIR/%08d.jpg" $OUPUT_FILE
ffmpeg -r 25 -qscale 2 -i "$SPLITTEDPROCESSEDDIR/%08d.jpg" $OUPUT_FILE  > /dev/null 2>&1 

