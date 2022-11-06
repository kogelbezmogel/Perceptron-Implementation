#ifndef __PERCEPTRON__
#define __PERCEPTRON__

#include <vector>
#include <iostream>
#include <functional>
#include <map>
#include <random>
#include <map>


typedef std::vector<double> Vec;
typedef std::vector< Vec > Mat;
typedef std::function< double (double x) > Function;
typedef std::function< double(double, double) > Function2args;
typedef std::map< std::string, Function > ActFunMap;
typedef std::map< std::string, Function > GradientFunMap;
typedef std::map< std::string, Function2args > LossFunMap;
typedef std::map< std::string, Function2args > LossGradMap;


class Perceptron {

    // declaration of nested class
    class FunsContainer;

    bool _empty = {false};
    Vec _weights;
    double _bias;
    Function _act_fun;
    Function _grad_act_fun;
    Function2args _grad_loss_fun;
    Function2args _loss_fun;
    
    static FunsContainer _acitivation_function_container;

    public:
        Perceptron();
        Perceptron(std::string activation_function, std::string loss_fun);
        void train(Mat x_train, Mat y_train);
        Mat predict(Mat x_test);

        void set_params(Vec new_params);
        void set_bias(double new_bias);

        Vec get_weights() const { return _weights; };
        double get_bias() const { return _bias; };

        void save( std::string path ) const;
        void load( std::string path );

    private:
        class FunsContainer {
            ActFunMap _act_fun_map;
            GradientFunMap _grad_fun_map;
            LossFunMap _loss_fun_map;
            LossGradMap _loss_grad_map;

            public:
                FunsContainer();
                Function get_act_fun(std::string name);
                Function get_grad_act_fun(std::string name);
                Function2args get_loss_fun(std::string name);
                Function2args get_gard_loss(std::string name);
        };
};


std::ostream& operator<< (std::ostream& str, const Perceptron& print);

#endif //__PERCEPTRON__
