#include <iostream>
#include "NeuralNetwork.h"



Pmap Model::initialize_parameters(const vector<int> &layer_dims, const vector<string> &activations){
    Pmap parameters;
    this->activations = activations;

    int size = layer_dims.size();
    for (int l = 1; l < size; l++){
        double epsilon = sqrt(2.0 / layer_dims[l-1]); // He initialization
        parameters["W" + to_string(l)] = MatrixXd::Random(layer_dims[l], layer_dims[l-1]) * epsilon;
        parameters["b" + to_string(l)] = VectorXd::Zero(layer_dims[l]);
    }
    this->parameters = parameters;
    return parameters;
}

MatrixXd Model::sigmoid(const MatrixXd &x) {
    return 1 / (1 + (-x).array().exp());
}

MatrixXd Model::relu(const MatrixXd &x) {
    return x.cwiseMax(0);
}

MatrixXd Model::tanh(const MatrixXd &x) {
    return x.array().tanh();
}

MatrixXd Model::leaky_relu(const MatrixXd &x, double alpha) {
    return x.cwiseMax(alpha * x);
}

MatrixXd Model::softmax(const MatrixXd &x) {
    return x.array().exp() / x.array().exp().sum();
}

tuple<MatrixXd, Cache_tuple> Model::Linear_Forward(const MatrixXd &A, const MatrixXd &W, const VectorXd &b, const string &activation){
    MatrixXd Z = W * A;
    Z.colwise() += b;
    MatrixXd A_next = A;

    if (activation_functions.find(activation) != activation_functions.end())
        A_next = activation_functions[activation](Z);
    else{
        cerr << "Activation function not found: " << activation << endl;
        exit(1);
    }
    
    Cache_tuple cache = make_tuple(A, W, b, Z);
    return make_tuple(A_next, cache);
}

Fw_tuple Model::Forward_Propagation(const MatrixXd &X, const Pmap &parameters){
    int L = parameters.size() / 2;
    vector<Cache_tuple> caches;

    MatrixXd A = X;
    MatrixXd W, Z;
    VectorXd b;
    string activation;
    
    for (int l = 1; l < L + 1 ; l++) {
        W = parameters.at("W" + to_string(l));
        b = parameters.at("b" + to_string(l));
        activation = activations[l-1];

        tuple<MatrixXd, Cache_tuple> result = Linear_Forward(A, W, b, activation);

        A = get<0>(result);
        caches.push_back(get<1>(result));
    }

    return make_tuple(A, caches);
}

double Model::calculate_loss(const MatrixXd &AL, const MatrixXd &Y, const string &loss_type){
    double m = AL.cols();
    double loss;
    
    if (loss_type == "binary_cross_entropy") {
        loss = -1.0 / m * (Y.array() * (AL.array().log()) + (1 - Y.array()) * ((1 - AL.array()).log())).sum();
    }else if (loss_type == "categorical_cross_entropy") {
        loss = -1.0 / m * (Y.array() * AL.array().log()).sum();
    }else if (loss_type == "mean_squared_error"){
        loss = 1.0 / m * (AL - Y).array().pow(2).sum();
    }else {
        cerr << "Unknown loss type: " << loss_type << endl;
        exit(1);
    }
    return loss;
}

MatrixXd Model::relu_backward(const MatrixXd &dA, const MatrixXd &Z){
    MatrixXd dZ = dA.array() * (Z.array() > 0).cast<double>();
    return dZ;
}

MatrixXd Model::sigmoid_backward(const MatrixXd &dA, const MatrixXd &Z){
    MatrixXd s = sigmoid(Z);
    MatrixXd dZ = dA.array() * s.array() * (1 - s.array());
    return dZ;
}

MatrixXd Model::tanh_backward(const MatrixXd &dA, const MatrixXd &Z){
    MatrixXd dZ = dA.array() * (1 - Z.array().tanh().array().pow(2));
    return dZ;
}

MatrixXd Model::leaky_relu_backward(const MatrixXd &dA, const MatrixXd &Z, double alpha){
    MatrixXd dZ = dA.array() * ((Z.array() > 0).cast<double>() + alpha * (Z.array() <= 0).cast<double>());
    return dZ;
}

MatrixXd Model::softmax_backward(const MatrixXd &dA, const MatrixXd &Z){
    MatrixXd A = softmax(Z);
    MatrixXd dZ = A - dA;

    return dZ;
}


Grad_tuple Model::Linear_Backward(const MatrixXd &dA, const Cache_tuple &cache, const string &activation){
    MatrixXd A, W, Z;
    VectorXd b;
    tie(A, W, b, Z) = cache;
    double m = A.cols();

    MatrixXd dZ, dW, dA_prev;
    VectorXd db;

    if (activation_derivatives.find(activation) != activation_derivatives.end())
        dZ = activation_derivatives[activation](dA, Z);
    else{
        cerr << "Activation function not found: " << activation << endl;
        exit(1);
    }
        

    dW = 1/m * (dZ * A.transpose()).array();
    db = 1/m * dZ.rowwise().sum();
    dA_prev = W.transpose() * dZ;

    return make_tuple(dA_prev, dW, db);
}

Gmap Model::BackwardPropagation(const MatrixXd &AL, MatrixXd Y, const vector<Cache_tuple> &caches, const string &loss_type){
    Gmap grads;
    int L = caches.size();
    double m = AL.cols();
    Y.resize(AL.rows(), AL.cols());

    MatrixXd dA;

    if (loss_type == "binary_cross_entropy"){
        dA = - (Y.array() / AL.array() - (1 - Y.array()) / (1 - AL.array()));

    }else if (loss_type == "categorical_cross_entropy") {
        if (activations[L - 1] == "softmax"){
            dA = AL.array() - Y.array();
        }
        else
            dA = - (Y.array() / AL.array());

    }else if (loss_type == "mean_squared_error"){
        dA = 2.0 / m * (AL - Y);

    }else {
        cerr << "Unknown loss type: " << loss_type << endl;
        exit(1);
    }

    for (int l = L; l > 0; l--){
        string activation = activations[l-1]; 
        Cache_tuple cache = caches[l-1];
        Grad_tuple result = Linear_Backward(dA, cache, activation);

        grads["dA" + to_string(l)] = get<0>(result);
        grads["dW" + to_string(l)] = get<1>(result);
        grads["db" + to_string(l)] = get<2>(result);

        dA = grads["dA" + to_string(l)];
    }
    return grads;

}

Model::Model() {
    activation_functions = {
        {"relu", std::bind(&Model::relu, this, std::placeholders::_1)},
        {"sigmoid", std::bind(&Model::sigmoid, this, std::placeholders::_1)},
        {"tanh", std::bind(&Model::tanh, this, std::placeholders::_1)},
        {"leaky_relu", std::bind(&Model::leaky_relu, this, std::placeholders::_1, 0.01)},
        {"softmax", std::bind(&Model::softmax, this, std::placeholders::_1)}
    };

    activation_derivatives = {
        {"relu", std::bind(&Model::relu_backward, this, std::placeholders::_1, std::placeholders::_2)},
        {"sigmoid", std::bind(&Model::sigmoid_backward, this, std::placeholders::_1, std::placeholders::_2)},
        {"tanh", std::bind(&Model::tanh_backward, this, std::placeholders::_1, std::placeholders::_2)},
        {"leaky_relu", std::bind(&Model::leaky_relu_backward, this, std::placeholders::_1, std::placeholders::_2, 0.01)},
        {"softmax", std::bind(&Model::softmax_backward, this, std::placeholders::_1, std::placeholders::_2)}
    };
}

Pmap Model::update_parameters(Pmap &parameters, Gmap &grads, const double &learning_rate){
    int L = parameters.size() / 2;
    for (int l = 1; l <= L; l++){
        parameters["W" + to_string(l)] -= learning_rate * grads["dW" + to_string(l)];
        parameters["b" + to_string(l)] -= learning_rate * grads["db" + to_string(l)];
    }

    return parameters;
}

void Model::train(const MatrixXd &X, const MatrixXd &Y, const vector<int> &layer_dims, const vector<string> &activations, const string &loss_type, const double &learning_rate, const int &num_iterations){
    this->layer_dims = layer_dims;
    this->parameters = initialize_parameters(layer_dims, activations);
    
    for (int i = 0; i <= num_iterations; i++){
        Fw_tuple result = Forward_Propagation(X, parameters);
        MatrixXd AL = get<0>(result);
        vector<Cache_tuple> caches = get<1>(result);

        double loss = calculate_loss(AL, Y, loss_type);

        Gmap grads = BackwardPropagation(AL, Y, caches, loss_type);

        parameters = update_parameters(parameters, grads, learning_rate);

        if (i % 1000 == 0)
            cout << "Cost after iteration " << i << ": " << loss << endl;
    }
}

MatrixXd Model::predict(const MatrixXd &X, const Pmap &parameters){
    Fw_tuple result = Forward_Propagation(X, parameters);
    MatrixXd prediction = get<0>(result);

    return prediction;
}

void Model::save_parameters(const string &filename) {
    ofstream file(filename);
    if (file.is_open()) {
        for (const auto& param : parameters) {
            file << param.first << "\n";
            file << param.second.rows() << "," << param.second.cols() << "\n";
            for (int i = 0; i < param.second.rows(); ++i) {
                for (int j = 0; j < param.second.cols(); ++j) {
                    file << param.second(i, j);
                    if (j < param.second.cols() - 1) file << ",";
                }
                file << "\n";
            }
            file << "\n";
        }
        file.close();
    } else 
        cerr << "Unable to open file for writing: " << filename << endl;
}

void Model::load_parameters(const string &filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Unable to open file for reading: " << filename << endl;
        return;
    }

    string line;
    while (getline(file, line)) {
        string key = line;
        getline(file, line);
        stringstream ss(line);
        int rows, cols;
        char comma;
        ss >> rows >> comma >> cols;

        MatrixXd param(rows, cols);
        for (int i = 0; i < rows; ++i) {
            getline(file, line);
            stringstream lineStream(line);
            for (int j = 0; j < cols; ++j) {
                string cell;
                getline(lineStream, cell, ',');
                param(i, j) = stod(cell);
            }
        }
        parameters[key] = param;
        getline(file, line); // Skip blank line
    }

    file.close();
}