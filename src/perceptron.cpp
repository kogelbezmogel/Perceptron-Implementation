#include <random>
#include <algorithm>
#include "../include/perceptron.h"

void createRandomOrder( VecData& order_vec, int length );

Perceptron::Perceptron() {
    _act_fun = [] (double x) { return ( x > 0 ? 1 : -1); }; 
    _loss_fun_gradient = [](std::vector<double> x, double y, std::vector<double> w, int weight_number) {
        
        double ret = 0;
        double Li = 0;

        for( int i = 0; i < x.size(); ++i ) //counting error value if positive then classification is correct
            Li += x[i] * w[i];
        Li += w[ w.size() - 1 ] * 1;
        Li *= -y;

        if( Li < 0 ) //not correct classification then gradient value is not zero
            if ( weight_number != w.size() - 1 )
                ret = -y * x[weight_number];
            else
                ret = -y * 1; 

        return ret; 
    };
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
    int amount_of_weigths = x_train[0].size() + 1; //index 0 for bias
    double max_error = 0.001;
    double error = 1;
    double expected_y;
    double calculated_y;
    double learning_const = 0.01;
    VecData order_vec;
    createRandomOrder( order_vec, x_train.size() );
    
    _w.resize(amount_of_weigths, 1);

    // need some additional step to check if net is configured becouse the _w and _bias 
    // might have been set by user before training  
    _w[0] = 1.0;
    _w[1] = 1.0;
    _w[2] = 1.0; //bias


    int i;
    for( int k = 0; k < epochs; ++k ) {

        for( int index = 0; index < train_amount; ++index ) {    
            i = order_vec[index]; //rondom order of inputs 
            for( int j = 0; j < amount_of_weigths; ++j ) {
                for( int ite = 0; ite < iterations_for_gradient_descent; ++ite ) {
                    expected_y = y_train[i];
                    _w[j] += learning_const * _loss_fun_gradient(x_train[i], expected_y, _w, j);
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
    str << "] \n";

    return str;
}