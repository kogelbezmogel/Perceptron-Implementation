#include <random>
#include <iostream>
#include <string>
#include <functional>

typedef std::function< double (double x) > Function;
typedef std::vector< std::vector<double> > Mat;

class DataGeneratorC {

    // fisrt set of distribution params
    double _m_x1;
    double _m_y1;
    double _q_x1;
    double _q_y1;

    // second set of params
    double _m_x2;
    double _m_y2;
    double _q_x2;
    double _q_y2;

    public:
        DataGeneratorC();
        void set_first_gruop_params(double m_x, double q_x, double m_y, double q_y);
        void set_second_group_params(double m_x, double q_x, double m_y, double q_y);

        void operator()(int k, std::string path = "", bool random=false);
};


class DataGeneratorR {
    Function _fun;

    public:
        DataGeneratorR(double a, double b);
        DataGeneratorR(Function fun);

        void operator()(double a, double b,  std::string path = "", bool random=false);
};