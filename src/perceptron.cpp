#include <random>
#include <algorithm>
#include <time.h>
#include <stdlib.h>
#include "../include/perceptron.h"

FunsContainer Perceptron::_funs = FunsContainer();
void createRandomOrder( VecData& order_vec, int length, std::mt19937& gen );
void createRandomOrder2(  VecData& order_vec, int length, std::mt19937& gen );

Perceptron::Perceptron() {
    std::random_device rd;
    _gen = std::mt19937( rd() );

    _act_fun = _funs.getActFun( "sign" ); 
    _loss_fun_gradient = _funs.getGradFun( "linearGradiend" );
 }

VecData Perceptron::operator() (MatData x_test) {

    //error detection check sizes for _w and x_test
    VecData result;
    double y;

    for( VecData& vec : x_test ) {
        y = 0;
        for( int i = 0; i < _w.size() - 1; ++i ) {
            y += _w[i] * vec[i];
        }
        y += _w[ _w.size() - 1]; //bias
        result.push_back( _act_fun(y) );
    }
    return result;
}


double Perceptron::operator() (VecData x_test) {

    //error detection check sizes for _w and x_test
    double result;

    result = 0;
    for( int i = 0; i < _w.size() - 1; ++i ) {
        result += _w[i] * x_test[i];
    }
    result +=_w[ _w.size() - 1 ]; //bias

return _act_fun( result );
}

void Perceptron::setActFun( std::string fun_name ) {

}

void Perceptron::printAvailableActFuns() {

}

void Perceptron::train(MatData x_train, VecData y_train) {
    
    //error detection check size x_train and y_train

    int epochs = 60;
    int iterations_for_gradient_descent = 4;
    int train_amount = x_train.size();
    int amount_of_weigths = x_train[0].size() + 1; //index 0 for bias
    double max_error = 0.0001;
    double error = 1;
    double expected_y;
    double calculated_y;
    double learning_const = 0.001;
    VecData order_vec = {0};
    
    _w.resize(amount_of_weigths, 1);

    // need some additional step to check if net is configured becouse the _w and _bias 
    // might have been set by user before training  
    _w[0] = 1.0;
    _w[1] = 1.0;
    _w[2] = 1.0; //bias


    int i;
    for( int k = 0; k < epochs; ++k ) {
        createRandomOrder2( order_vec, train_amount, this -> _gen );
        for( int index = 0; index < train_amount; ++index ) {    
            i = order_vec[index]; //random order of inputs 
            for( int j = 0; j < amount_of_weigths; ++j ) {
                for( int ite = 0; ite < iterations_for_gradient_descent; ++ite ) {
                    expected_y = y_train[i];
                    _w[j] -= learning_const * _loss_fun_gradient(x_train[i], expected_y, _w, j);
                }
            }
        }
    }
}

void createRandomOrder( VecData& order_vec, int length, std::mt19937& gen ) {
    order_vec.resize(length, 0);
    for( int i = 0; i < length - 1; ++i )
        order_vec[i] = i;

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

void createRandomOrder2(  VecData& order_vec, int length, std::mt19937& gen ) {
    srand( time(NULL) );
    order_vec.resize(length, 0);
    for( int i = 0; i < length - 1; ++i )
        order_vec[i] = i;

    int temp;
    int rand_index;
    
    for( int i = 0; i < length - 1; ++i ) {
      rand_index = rand() % (length - i) + i;

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