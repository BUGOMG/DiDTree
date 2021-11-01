#ifndef _dataset_H
#define _dataset_H

# include <armadillo>
#include <fstream>
#include <memory>

# include "types.h"


using namespace std;

namespace didtree
{
class Dataset{
    std::shared_ptr<RMat> p_data;
    std::shared_ptr<RMat> p_targets;
    std::shared_ptr<RMat> p_treatments;

    arma::field<string> head_data;
    arma::field<string> head_targets;
    arma::field<string> head_treatments;

    size_t n_instances;
    size_t n_features;
    size_t n_treatments;
    size_t n_targets;
    size_t n_periods;
    size_t n_treated;
public:
    Dataset(const std::string &path, size_t n_treatments=2, size_t n_treated=1);

    void descript()const;
};
} // namespace didtree
#endif //_dataset_H