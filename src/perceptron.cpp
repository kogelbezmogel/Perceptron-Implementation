#include <random>
#include <algorithm>
#include "../include/perceptron.h"

void createRandomOrder( VecData& order_vec, int length );

Perceptron::Perceptron() {
    _act_fun = [] (double x) { return ( x > 0 ? 1 : -1); }; 
    _loss_fun_gradient = [](double x, double y, double w) { return std::max( -y * x * w, 0.0); };
    //_act_fun = [] (double x) { return x; };
 }

VecData Perceptron::operator() (MatData x_test) {

    //error detection check sizes for _w and x_test
    VecData result;
    double y;

    for( VecData& vec : x_test ) {

        y = 0;
        for( int i = 0; i < _w.size(); ++i ) {
            y += _w[i] * vec[i];
        }
        y += _bias;
        result.push_back( _act_fun(y) );
    }
    return result;
}



double Perceptron::operator() (VecData x_test) {

    //error detection check sizes for _w and x_test
    double result;

    result = 0;
    for( int i = 0; i < _w.size(); ++i ) {
        result += _w[i] * x_test[i];
    }
    result += _bias;

return _act_fun( result );
}

void Perceptron::setActFun( std::string fun_name ) {

}

void Perceptron::setLossFun( std::string fun_name ) {

}

void Perceptron::printAvailableActFuns() {

}

void Perceptron::printAvailableLossFuns() {

}

void Perceptron::train(MatData x_train, VecData y_train) {
    
    //error detection check size x_train and y_train

    int epochs = 100;
    int iterations_for_gradient_descent = 20;
    int train_amount = x_train.size();
    int perc_inputs = x_train[0].size();
    int amount_of_wages = perc_inputs;
    double max_error = 0.001;
    double error = 1;
    double expected_y;
    double calculated_y;
    double learning_const = 0.001;
    VecData order_vec;
    createRandomOrder( order_vec, x_train.size() );
    
    _w.resize(amount_of_wages, 1);

    // need some additional step to check if net is configured becouse the _w and _bias 
    // might have been set by user before training  
    _bias = 1.0;
    _w[0] = 1.0;
    _w[1] = 1.0;

    int i;
    for( int k = 0; k < epochs; ++k ) {
        for( int index = 0; index < train_amount; ++index ) {    
            i = order_vec[index]; //rondom order of inputs 
            for( int j = 0; j < perc_inputs; ++j ) {
                for( int ite = 0; ite < iterations_for_gradient_descent; ++ite ) {
                    expected_y = y_train[i];
                    calculated_y = (*this) ( x_train[i] );
                    error = expected_y - calculated_y;
                    
                    _w[j] += learning_const * error * _loss_fun_gradient(x_train[i][j], expected_y, _w[j]);
                    _bias += error * learning_const;
                }
            }
        }
    }
}

void createRandomOrder( VecData& order_vec, int length ) {
    order_vec.resize(length, 0);
    for( int i = 0; i < length - 1; ++i )
        order_vec[i] = i;

    std::random_device rd;
    std::mt19937 gen( rd() );
    int temp;
    int rand_index;
    
    for( int i = 0; i < length; ++i ) {
      std::uniform_int_distribution<> distrib(i, length - 1);
      rand_index = distrib( gen );

      temp = order_vec[i];
      order_vec[i] = order_vec[rand_index];
      order_vec[rand_index] = temp;
    }
}


std::ostream& operator<< (std::ostream& str, const Perceptron& per) {

    str << "wages: [ ";
    for( double w : per.weights() )
        str << w << " ";
    str << "]  bias: " << per.bias() << "\n";

    return str;
}