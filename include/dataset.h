#ifndef _dataset_H
#define _dataset_H

# include <armadillo>
#include <fstream>
#include <memory>

# include "types.h"


using namespace arma;
using namespace std;

namespace didtree
{
class Dataset{
    std::shared_ptr<RMat> p_data;
    std::shared_ptr<RMat> p_targets;
    std::shared_ptr<RMat> p_treatments;

    field<string> head_data;
    field<string> head_targets;
    field<string> head_treatments;
public:
    Dataset(const std::string & path);
};
} // namespace didtree
#endif //_dataset_H