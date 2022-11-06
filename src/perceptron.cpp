#include <random>
#include <algorithm>
#include <time.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <chrono>

#include "../include/perceptron.h"
#include "../include/activation_functions.h"

Perceptron::Perceptron() : _empty(true) { }

Perceptron::Perceptron(std::string activation_function, std::string loss_fun) {
    _act_fun = _acitivation_function_container.get_act_fun(activation_function);
    _grad_act_fun = _acitivation_function_container.get_grad_act_fun(activation_function);
    _loss_fun = _acitivation_function_container.get_loss_fun(loss_fun);
    _grad_loss_fun = _acitivation_function_container.get_gard_loss(loss_fun);
}


void Perceptron::train(Mat x_train, Mat y_train) {

    if( x_train.size() != y_train.size() ) // Place for exception
        std::cout << "ERROR! x_train y_train must me same size! " << x_train.size() << " vs " << y_train.size() << "\n";

    int batch_size = 128;
    int epochs = 500000;
    int data_size = y_train.size();
    double y_res = 0;
    double z_res = 0;
    double y_real;
    double alpha = 0.000001;
    
    _weights.resize(x_train[0].size(), 0);
    _bias = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < epochs; i++) {

        // training iterations
        for(int j = 0; j < data_size; j++) {
            
            // current prediction
            z_res = 0; 
            for(int k = 0; k < _weights.size(); k++) // add vector multiply inline
                z_res += _weights[k] * x_train[j][k];
            z_res += _bias;
            y_res = _act_fun(z_res);
            y_real = y_train[j][0];

            // updating weights
            for(int k = 0; k < _weights.size(); k++) {
                _weights[k] += alpha * _grad_loss_fun(y_real, y_res) * _grad_act_fun(z_res) * x_train[j][k]; 
            }
            _bias += alpha * _grad_loss_fun(y_real, y_res) * _grad_act_fun(z_res);
        }
        if( i % 1000 == 0 ) {
            auto end = std::chrono::high_resolution_clock::now();
            std::cout << "epoch: " << i << "  |time from start: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "\n";
        }
    }

    
    //final prediction error
    double pred_error = 0;
    for(int j = 0; j < data_size; j++) {
        z_res = 0; 
        for(int k = 0; k < _weights.size(); k++) // add vector multiply inline
            z_res += _weights[k] * x_train[j][k];
        z_res += _bias;
        y_res = _act_fun(z_res);
        y_real = y_train[j][0];
        pred_error += _loss_fun(y_real, y_res);
    }
    pred_error /= data_size;
    std::cout << "loss: " << pred_error << "\n";
    
}


Mat Perceptron::predict(Mat x_test) {
    Mat result;
    return result;
}


void Perceptron::set_params(Vec new_params) {
    _weights = new_params;
}


void Perceptron::set_bias(double new_bias) {
    _bias = new_bias;
}


void Perceptron::save(std::string path) const {
    std::ofstream file;
    file.open(path);
    for( auto w : _weights )
        file << w << " ";
    file << "\n";
    file << _bias << "\n";
    file.close();
}


void Perceptron::load(std::string path) {

    std::string line;
    std::string word;
    Vec weights;
    double bias;

    std::ifstream file;
    file.open(path);

    getline(file, line);
    std::stringstream ss(line);
    while ( getline(ss, word, ' ') ) 
        weights.push_back( std::stof(word) );
    
    getline(file, line);
    bias = std::stof(line);
    file.close();

    _weights = weights;
    _bias = bias;
    _empty = false;
}


std::ostream& operator<< (std::ostream& str, const Perceptron& per) {
    str << "----------Perceptron------------\n";
    str << "weghts: ";
    Vec weights = per.get_weights();
    for( double w : weights )
        str << w << " ";
    str << "\n";
    str << "bias:   " << per.get_bias() << "\n";
    str << "----------          ------------\n";

    return str;
}




//nested class definitions ------------------------------------------||-------------------------

Perceptron::FunsContainer Perceptron::_acitivation_function_container;

// constructor for nested class constuctor
Perceptron::FunsContainer::FunsContainer() {
    _act_fun_map.insert( {"linear", a_f::linearFun} );
    _act_fun_map.insert( {"sigmoid", a_f::sigmoidFun} );

    _grad_fun_map.insert( {"linear", a_f::linearFunGradient} );
    _grad_fun_map.insert( {"sigmoid", a_f::sigmoidGradient} ); 
    
    _loss_fun_map.insert( {"mse", a_f::mse} );
    _loss_fun_map.insert( {"binary-crossentropy", a_f::binary_crossentropy} );

    _loss_grad_map.insert( {"binary-crossentropy", a_f::binary_crossentropyGradient} );
    _loss_grad_map.insert( {"mse", a_f::mseGradient} );
}

Function Perceptron::FunsContainer::get_act_fun(std::string name) {
    return _act_fun_map[name];
}

Function Perceptron::FunsContainer::get_grad_act_fun(std::string name) {
    return _grad_fun_map[name];
}

Function2args Perceptron::FunsContainer::get_loss_fun(std::string name) {
    return _loss_fun_map[name];
}  

Function2args Perceptron::FunsContainer::get_gard_loss(std::string name) {
    return _loss_grad_map[name];
}