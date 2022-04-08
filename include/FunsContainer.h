#ifndef __FUNS_CONTAINER__
#define __FUNS_CONATINER__

#include <functional>
#include <vector>
#include <map>
#include <string>
#include <utility>

typedef std::function< double (std::vector<double> X, double y, std::vector<double> W, int  weight_number) > GradientFun;
typedef std::function< double (double x) > ActivationFun;
typedef std::map< std::string, ActivationFun > ActFunMap;
typedef std::map< std::string, GradientFun > GradientFunMap;

struct FunsContainer {

    FunsContainer();
    ActivationFun getActFun( std::string name ) { return _act_fun_map.find(name) -> second; }
    GradientFun getGradFun( std::string name ) { return _grad_fun_map.find(name) -> second; }

    private:
        ActFunMap _act_fun_map;
        GradientFunMap _grad_fun_map;
};

#endif //__FUNS_CONTAINER__