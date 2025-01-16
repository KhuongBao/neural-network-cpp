EIGEN_PATH = "[PATH_TO_EIGEN]"

SOURCE = NeuralNetwork.cpp
FILE = example.cpp
OUT = example

# Default target
all: run

run: $(SOURCE) $(FILE)
	g++ -I $(EIGEN_PATH) $(SOURCE) $(FILE) -o $(OUT) && .\$(OUT) && del $(OUT).exe

