#ifndef __PERCEPTRON__
#define __PERCEPTRON__

#include "FunsContainer.h"
#include <vector>
#include <iostream>
#include <functional>
#include <map>

typedef std::vector< std::vector<double> > MatData;
typedef std::vector<double> VecData;
typedef std::map< std::string, std::function< double(double)> > MappedFuns;

class Perceptron {

    private:
        static FunsContainer _funs;
        std::vector<double> _w;
        std::function< double(double) > _act_fun;
        std::function< double( std::vector<double>, double, std::vector<double>, int ) > _loss_fun_gradient;

    public:
        Perceptron();

        std::function< double(double) >& activationFunction() { return _act_fun; };

        std::vector<double>& weights() { return _w; };
        std::vector<double> weights() const { return _w; };

        double& bias() { return _w[ _w.size() - 1 ]; };
        double bias() const { return _w[ _w.size() - 1 ]; };

        void setActFun( std::string fun_name );
        void setLossFun( std::string fun_name );
        void printAvailableActFuns();
        void printAvailableLossFuns();

        void train(MatData x_train, VecData y_train);

        VecData operator() ( MatData x_test );
        double operator() ( VecData x_test );
};

std::ostream& operator<< (std::ostream& str, const Perceptron& print);

#endif //__PERCEPTRON__
