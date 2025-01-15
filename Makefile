EIGEN_PATH = "D:/Code/C++/toolbox/eigen-3.4.0"

SOURCE = NeuralNetwork.cpp
FILE = example.cpp
OUT = example

# Default target
all: run

run: $(SOURCE) $(FILE)
	g++ -I $(EIGEN_PATH) $(SOURCE) $(FILE) -o $(OUT) && .\$(OUT) && del $(OUT).exe

