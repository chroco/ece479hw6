#!/bin/bash
CANNY=../src/canny
HOUGH=../src/houghdemo
k=0
for i in doors/*
do
	name="doors/door${k}.jpg"
#  mv "$i" ${name}
#	convert ${name} -resize 50x50 -gravity center -background "rgb(0,0,0)" -extent 50x50 ${name}
	$CANNY $i
	$HOUGH $i
#	k=$((k+1))
done
