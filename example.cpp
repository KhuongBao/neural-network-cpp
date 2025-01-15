#include <iostream>
#include "NeuralNetwork.h"

using namespace std;

MatrixXd readCSV(const string &file, int rows, int cols);
void save_predictions(const MatrixXd &predictions, const string &filename);


int main(){
    MatrixXd train_data = readCSV("train.csv", 300, 3);
    MatrixXd X = train_data.block(0, 0, train_data.rows(), 2).transpose();
    MatrixXd Y = train_data.block(0, 2, train_data.rows(), 1).transpose();
    MatrixXd prediction;

    // Model consisting of 1 input layer, 2 hidden layers with 20 and 5 units, and 1 output layer
    vector<int> layer_dims = {X.rows(), 20, 5, 1};
    vector<string> activations = {"relu", "relu", "sigmoid"};

    Model model;
    double learning_rate = 0.01;
    double num_iterations = 20000;

    model.train(X, Y, layer_dims, activations, "binary_cross_entropy", learning_rate, num_iterations);
    model.save_parameters("parameters.csv");
    // model.load_parameters("parameters.csv");

    prediction = model.predict(X, model.parameters);
    save_predictions(prediction, "predictions_train.csv");


    MatrixXd test_data = readCSV("test.csv", 100, 3);
    MatrixXd X_test = test_data.block(0, 0, test_data.rows(), 2).transpose();

    prediction = model.predict(X_test, model.parameters);
    save_predictions(prediction, "predictions_test.csv");

    return 0;
}


// Function to read CSV file into Eigen matrix
MatrixXd readCSV(const string &file, int rows, int cols) {
    ifstream in(file);
    string line;
    MatrixXd data(rows, cols);
    int row = 0;
    while (getline(in, line)) {
        stringstream lineStream(line);
        string cell;
        int col = 0;
        while (getline(lineStream, cell, ',')) {
            data(row, col) = stod(cell);
            col++;
        }
        row++;
    }
    return data;
}

void save_predictions(const MatrixXd &predictions, const string &filename) {
    ofstream file(filename);
    if (file.is_open()) {
        file << predictions.rows() << "," << predictions.cols() << "\n";
        for (int i = 0; i < predictions.rows(); ++i) {
            for (int j = 0; j < predictions.cols(); ++j) {
                file << predictions(i, j);
                if (j < predictions.cols() - 1) {
                    file << ",";
                }
            }
            file << "\n";
        }
        file.close();
        cout << "Predictions saved to " << filename << endl;
    } else {
        cerr << "Unable to open file for writing: " << filename << endl;
    }
}