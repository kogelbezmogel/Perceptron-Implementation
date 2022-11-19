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


void Perceptron::train(Mat x_train, Mat y_train, int epochs, int batch_size, std::string optimazer) {

    if( x_train.size() != y_train.size() ) // Place for exception
        std::cout << "ERROR! x_train y_train must me same size! " << x_train.size() << " vs " << y_train.size() << "\n";

    int data_size = y_train.size();
    double y_res = 0;
    double z_res = 0;
    double y_real;
    double alpha = 0.9; //starting value of alpha
    double epsilon = 10e-7;
    int n_gradients = 5;

    // seting wieghts and bias to 0
    // this need to be changed in future
    _weights.resize(x_train[0].size(), 0);
    _bias = 0;


    // Adagrad learning rate algorithm variables
    // thoose are being used by AdaDelta
    Vec gradient_sq_sum;
    double bias_gradient_sq_sum = 0;
    gradient_sq_sum.resize(x_train[0].size(), epsilon);

    // AdaDelta learning rate algorithm variables
    int amount_of_gradients = 50;
    VecLists last_n_gradients_sq;
    std::list<double> last_n_bias_gradients_sq;
    last_n_gradients_sq.resize( _weights.size() );
    for( std::list<double>& list : last_n_gradients_sq )
        list.resize(amount_of_gradients, 0);
    last_n_bias_gradients_sq.resize(amount_of_gradients, 0);

    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < epochs; i++) {

        int n_batches = std::ceil( (double) data_size / batch_size );
        int b_start;
        int b_end;
        int N;
  
        // counting gradients in batch 
        for(int b = 0; b < n_batches; b++) {
            b_start = b * batch_size;
            b_end = std::min( (b+1)*batch_size, data_size );
            N = b_end - b_start;

            // counting sum of gradients over samples in batch
            Vec gradient_avg;
            double bias_avg = 0;
            gradient_avg.resize(_weights.size(), 0);

            for(int row = b_start; row < b_end; row++) {

                // current prediction
                z_res = 0; 
                for(int k = 0; k < _weights.size(); k++) // add vector multiply inline
                    z_res += _weights[k] * x_train[row][k];
                z_res += _bias;
                y_res = _act_fun(z_res);
                y_real = y_train[row][0];

                for(int w = 0; w < gradient_avg.size(); w++) {
                    gradient_avg[w] +=  _grad_loss_fun(y_real, y_res) * _grad_act_fun(z_res) * x_train[row][w];
                    bias_avg +=  _grad_loss_fun(y_real, y_res) * _grad_act_fun(z_res);
                }
            }

            // counting mean of gradients in batch
            for(int w = 0; w < _weights.size(); w++) 
                gradient_avg[w] = gradient_avg[w] / N;
            bias_avg = bias_avg / N;
            
            if(optimazer == "AdaGrad") {
                for(int w = 0; w < _weights.size(); w++)
                    gradient_sq_sum[w] += std::pow(gradient_avg[w], 2);
                bias_gradient_sq_sum += std::pow(bias_avg, 2);

                // updating weights with average gradient over batch for AdaGrad algorithm
                for(int w = 0; w < _weights.size(); w++) 
                    _weights[w] += (alpha / std::sqrt(gradient_sq_sum[w])) * gradient_avg[w];
                _bias += (alpha / std::sqrt(bias_gradient_sq_sum)) * bias_avg;
            }

            if(optimazer == "AdaDelta") {
                //updating lists of gradients and sums to include only last 5 elements
                for( int w = 0;  w < _weights.size(); w++ ) {
                    last_n_gradients_sq[w].push_front( std::pow(gradient_avg[w],2) );
                    gradient_sq_sum[w] = gradient_sq_sum[w] + last_n_gradients_sq[w].front() - last_n_gradients_sq[w].back();
                    last_n_gradients_sq[w].pop_back();
                }
                last_n_bias_gradients_sq.push_front( std::pow(bias_avg,2) );
                bias_gradient_sq_sum = bias_gradient_sq_sum + last_n_bias_gradients_sq.front() - last_n_bias_gradients_sq.back();
                last_n_bias_gradients_sq.pop_back();

                // updating weights with average gradient over batch for AdaGrad algorithm
                for(int w = 0; w < _weights.size(); w++)
                    _weights[w] += (alpha / std::sqrt(gradient_sq_sum[w])) * gradient_avg[w];
                _bias += (alpha / std::sqrt(bias_gradient_sq_sum)) * bias_avg;
            }

            if(optimazer == "Adam") {

            }
    
        }

        if( i % 1 == 0 ) {
            auto end = std::chrono::high_resolution_clock::now();
            double pred_error = 0;
            for(int j = 0; j < data_size; j++) {
                z_res = 0; 
                for(int k = 0; k < _weights.size(); k++) // add vector multiply inline
                    z_res += _weights[k] * x_train[j][k];
                z_res += _bias;
                y_res = _act_fun(z_res);
                y_real = y_train[j][0];
                if( std::isnan((y_real, y_res)) ) {
                    std::cout << "z_res: " << z_res << "\n";
                    std::cout << "y_res: " << y_res << "\n";
                    exit(0);
                }
                pred_error += _loss_fun(y_real, y_res);
            }
            pred_error /= data_size;
            std::cout << "epoch: " 
                      << i 
                      << "  |time from start: "
                      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
                      << " | loss: "
                      << pred_error
                      << " | alphas coeffs: ";
            for(double sum : gradient_sq_sum)
                std::cout << alpha / std::sqrt(sum) << "  ";
            std::cout << "| bias coeff: " << alpha / std::sqrt(bias_gradient_sq_sum) << "\n";
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