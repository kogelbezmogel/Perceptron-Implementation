#ifndef __PERCEPTRON__
#define __PERCEPTRON__

#include <vector>
#include <iostream>
#include <functional>

typedef std::vector< std::vector<double> > MatData; 
typedef std::vector<double> VecData;

class Perceptron {

    private:
        std::vector<double> _w;
        std::function< double(double) > _act_fun;
        double _bias;
    
    public:
        Perceptron();

        std::function< double(double) >& activationFunction() { return _act_fun; };
        std::vector<double>& wages() { return _w; };
        std::vector<double> wages() const { return _w; };
        double& bias() { return _bias; };
        double bias() const { return _bias; };

        void train(MatData x_train, VecData y_train);

        VecData operator() ( MatData x_test );
        double operator() ( VecData x_test );
};

std::ostream& operator<< (std::ostream& str, const Perceptron& print);

#endif //__PERCEPTRON__