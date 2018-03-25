#!/bin/bash
k=0
for i in doors/*
do
  mv "$i" "doors/door${k}.jpg"
	k=$((k+1))
#	echo $k
done
