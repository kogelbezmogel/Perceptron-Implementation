#include "../include/data_generation.h"
#include <ctime>
#include <fstream>


DataGeneratorC::DataGeneratorC() {

}

void DataGeneratorC::set_first_gruop_params(double m_x, double q_x, double m_y, double q_y) {
    _m_x1 = m_x;
    _m_y1 = m_y;
    _q_x1 = q_x;
    _q_y1 = q_y;
}

void DataGeneratorC::set_second_group_params(double m_x, double q_x, double m_y, double q_y) {
    _m_x2 = m_x;
    _m_y2 = m_y;
    _q_x2 = q_x;
    _q_y2 = q_y;

}

void DataGeneratorC::operator() (int k, std::string path, bool random) {
    //std::cout << "I am creating some random distribution: \n";

    // wybranie silnika i nakarmienie go seedem
    std::mt19937 engine;
    int s = std::time(0);
    //std::cout << "Seed: " << s << "\n";
    engine.seed(s);

    std::normal_distribution<double> x_dist1(_m_x1, _q_x1);  //(mean, std_devation)
    std::normal_distribution<double> y_dist1(_m_y1, _q_y1);

    std::vector< std::pair<double, double> > points_1;
    for(int i = 0; i < k; i++)
        points_1.push_back( std::pair<double, double>(x_dist1(engine), y_dist1(engine)) );

    std::normal_distribution<double> x_dist2(_m_x2,  _q_x2);
    std::normal_distribution<double> y_dist2(_m_y2, _q_y2);
    std::vector< std::pair<double, double> > points_2;
    for(int i = 0; i < k; i++)
        points_2.push_back( std::pair<double, double>(x_dist2(engine), y_dist2(engine)) );

    if( path == "" )
        path = "./Cdata1.csv";

    
    std::ofstream file;
    file.open(path);
    file << "X,Y,class\n";
    for(auto p : points_1)
        file << p.first << "," << p.second << "," << "0\n";
    for(auto p : points_2)
        file << p.first << "," << p.second << "," << "1\n";
    file.close();
}




DataGeneratorR::DataGeneratorR(double a, double b) {
    _fun = Function( [a, b](double x) { return a * x + b; } );
}

DataGeneratorR::DataGeneratorR(Function fun) {
    _fun = fun;
}

void DataGeneratorR::operator() (double a, double b, std::string path, bool random) {

    // the simplest example of linear regression

    std::mt19937 engine;
    int s = std::time(0);
    engine.seed(s);

    int amount = 5000;
    std::normal_distribution<double> n_dist(0, 1);
    Mat data;

    double pointer = a;
    double delta = (b-a) / amount; 
    double y;
    double x;
    for(int i = 0; i < amount; i++) {
        x = i*delta;
        y = _fun(x) + n_dist(engine);
        data.push_back( std::vector<double>({x, y}) );
    }

    if( path == "" ) // saving file (convert to method)
    path = "./Rdata1.csv";
    
    std::ofstream file;
    file.open(path);
    file << "X,Y\n";
    for( auto row : data )
        file << row[0] << "," << row[1] << "\n";
    file.close();
}