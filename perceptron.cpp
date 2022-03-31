#include "perceptron.h"


Perceptron::Perceptron() {
    //_act_fun = [] (double x) { return ( x > 0 ? 1 : -1); }; 
    _act_fun = [] (double x) { return x; };
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



void Perceptron::train(MatData x_train, VecData y_train) {
    
    //error detection check size x_train and y_train

    int epochs = 300;
    double max_error = 0.001;
    double error = 1;
    double expected_y;
    double calculated_y;
    double learning_const = 0.01;
    int amount_of_wages = x_train[0].size();

    // need some additional step to check if net is configured becouse the _w and _bias 
    // might have been set by user before training
    _w.resize(amount_of_wages, 1);
   
    _bias = 1.0;
    _w[0] = 1.0;
    _w[1] = 1.0;

    //add few iteration for every entry in one epoch
    for( int k = 0; k < epochs; ++k ) {
        for( int i = 0; i < x_train.size(); ++i ) {    
            expected_y = y_train[i];
            calculated_y = (*this) ( x_train[i] );
            error = expected_y - calculated_y;

            for( int j = 0; j < _w.size(); ++j ) {
                _w[j] += learning_const * error * x_train[i][j]; 
            }
            _bias += error * learning_const;
        }
        std::cout << (*this);
    }
}


std::ostream& operator<< (std::ostream& str, const Perceptron& per) {

    str << "wages: [ ";
    for( double w : per.wages() )
        str << w << " ";
    str << "]  bias: " << per.bias() << "\n";

    return str;
}