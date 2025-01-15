#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <Eigen/Dense>
#include <map>
#include <tuple>
#include <vector>
#include <iomanip> 
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <functional>

using namespace std;
using namespace Eigen;

typedef map<string, MatrixXd> Pmap;
typedef map<string, MatrixXd> Gmap;
typedef tuple<MatrixXd, MatrixXd, VectorXd, MatrixXd> Cache_tuple; // A, W, b, Z
typedef tuple<MatrixXd, vector<Cache_tuple>> Fw_tuple; // AL, caches
typedef tuple<MatrixXd, MatrixXd, MatrixXd> Grad_tuple; // dA, dW, db

class Model {
public:
    Model();
    MatrixXd X, Y;
    vector<int> layer_dims;
    Pmap parameters;
    vector<string> activations;

    void train(const MatrixXd &X, const MatrixXd &Y, const vector<int> &layer_dims, const vector<string> &activations, const string &loss_type, const double &learning_rate = 0.01, const int &num_iterations = 10000);
    MatrixXd predict(const MatrixXd &X, const Pmap &parameters);
    void save_parameters(const string &filename);
    void load_parameters(const string &filename);

private:
    Pmap initialize_parameters(const vector<int> &layer_dims, const vector<string> &activations);

    MatrixXd sigmoid(const MatrixXd &x);
    MatrixXd relu(const MatrixXd &x);
    MatrixXd tanh(const MatrixXd &x);
    MatrixXd leaky_relu(const MatrixXd &x, const double alpha = 0.01);
    MatrixXd softmax(const MatrixXd &x);

    tuple<MatrixXd, Cache_tuple> Linear_Forward(const MatrixXd &A, const MatrixXd &W, const VectorXd &b, const string &activation);
    Fw_tuple Forward_Propagation(const MatrixXd &X, const Pmap &parameters);

    double calculate_loss(const MatrixXd &AL, const MatrixXd &Y, const string &loss_type);

    MatrixXd relu_backward(const MatrixXd &dA, const MatrixXd &Z);
    MatrixXd sigmoid_backward(const MatrixXd &dA, const MatrixXd &Z);
    MatrixXd tanh_backward(const MatrixXd &dA, const MatrixXd &Z);
    MatrixXd leaky_relu_backward(const MatrixXd &dA, const MatrixXd &Z, const double alpha = 0.01);
    MatrixXd softmax_backward(const MatrixXd &dA, const MatrixXd &Z);

    Grad_tuple Linear_Backward(const MatrixXd &dA, const Cache_tuple &cache, const string &activation);
    Gmap BackwardPropagation(const MatrixXd &AL, MatrixXd Y, const vector<Cache_tuple> &caches, const string &loss_type);

    unordered_map<string, function<MatrixXd(const MatrixXd&)>> activation_functions;
    unordered_map<string, function<MatrixXd(const MatrixXd&, const MatrixXd&)>> activation_derivatives; 

    Pmap update_parameters(Pmap &parameters, Gmap &grads, const double &learning_rate);
};

#endif