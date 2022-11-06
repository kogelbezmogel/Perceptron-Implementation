#ifndef __ACT_FUNS__
#define __ACT_FUNS__

#include <cmath>

// z = w_1 * x_1 + w_2 * x_2 + ... w_n * x_n + bias
// activation function doesn't know about wieghts and bias
namespace a_f {


    //-------------- activation funs
    inline double linearFun( double z) {
        return z;
    }

    inline double sigmoidFun( double z ) {
        return  1.0 / (1.0 + std::exp(-z));
    }

    //---------------- loss funs
    inline double mse(double y_real, double y_pred) {
        return 0.5 * std::pow(y_pred - y_real, 2);
    }

    inline double binary_crossentropy(double y_real, double y_pred) {
        return  -y_real * std::log10(y_pred) - (1 - y_real) * std::log10(1 - y_pred);
    }


    //---------------- gradients
    inline double linearFunGradient( double z ) {  
        return 1.0;
    }

    inline double sigmoidGradient( double z ) {
        return sigmoidFun(z) * (1 - sigmoidFun(z)); 
    }

    inline double mseGradient(double y_real, double y_pred) {
        return (y_real - y_pred);
    }

    inline double binary_crossentropyGradient(double y_real, double y_pred) {
        return (y_real - y_pred) / ((y_pred - std::pow(y_pred, 2)) * std::log(10));
    }


}
#endif //__ACT_FUNS__