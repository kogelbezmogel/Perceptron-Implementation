#include <vector>
#include <iostream>

typedef std::vector<double> Vec;
typedef std::vector< Vec > Mat;

class DataFrame {
    bool _empty = {true};
    Mat _data;
    std::vector< std::string > _columns;

    public:
        void read_csv( std::string path );
        Vec& operator[] (std::string);
        Vec operator[] (int row_id);

};

std::ostream& operator<< (std::ostream& str, const DataFrame& d_frame);