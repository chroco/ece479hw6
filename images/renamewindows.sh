#!/bin/bash
k=0
for i in windows/*
do
  mv "$i" "windows/window${k}.jpg"
	k=$((k+1))
#	echo $k
done
