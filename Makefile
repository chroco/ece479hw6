CC=g++
#TARGET = canny
#SRC = CannyDetector_Demo.cpp
CFLAGS=-ggdb
FLAGS=`pkg-config --cflags --libs opencv`

all: canny houghdemo houghlines

#$(TARGET): $(SRC)
#	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(FLAGS) 

canny: CannyDetector_Demo.cpp
	$(CC) $(CFLAGS) CannyDetector_Demo.cpp -o canny $(FLAGS) 

houghdemo: HoughLines_Demo.cpp
	$(CC) $(CFLAGS) HoughLines_Demo.cpp -o houghdemo $(FLAGS) 

houghlines: HoughLines_Demo.cpp
	$(CC) $(CFLAGS) houghlines.cpp -o houghlines $(FLAGS) 

clean:
	rm canny houghdemo houghlines 
