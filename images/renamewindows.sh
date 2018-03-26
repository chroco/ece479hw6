#!/bin/bash
k=0
for i in windows/*
do
	name="windows/window${k}.jpg"
  mv "$i" ${name}
#	convert ${name} -resize 50x50 ${name}
	convert ${name} -resize 50x50 -gravity center -background "rgb(0,0,0)" -extent 50x50 ${name}
	k=$((k+1))
done
echo $k
