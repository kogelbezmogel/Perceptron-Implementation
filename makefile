all: compile link clean

compile:
	g++ -std=c++17 -c -O3 src/main.cpp src/data_generation.cpp src/perceptron.cpp

link:
	g++ -std=c++17 main.o data_generation.o perceptron.o -o main

clean:
	rm -f *.o