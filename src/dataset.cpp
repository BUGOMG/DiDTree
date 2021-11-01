#include <armadillo>
#include <iostream>
#include <string>
#include <memory>

#include "dataset.h"
#include "log.h"

didtree::Dataset::Dataset(const std::string &path, size_t n_treatments, size_t n_treated)
{
    field<std::string> hdata, htarget, htreatment;
    auto units = RMat();
    auto series = RMat();
    auto features = RMat();

    units.load(csv_name(path + "\\features.csv", hdata));
    series.load(csv_name(path + "\\targets.csv", htarget));
    BOOST_LOG_SEV(didtree::lg, didtree::INFO) << "has load " << hdata.n_cols << " lines data" << std::endl;    
    RMat data = arma::join_rows(units, series.rows(1, series.n_rows-1));
    std::cout<<data.n_cols<<data.n_rows<<std::endl;
    size_t feat_ist = 0, feat_num = units.n_cols, treat_ist = units.n_cols - 1, treat_num = 1;
    if (hdata[0].length() == 0)
    { //first line is the index column
        feat_ist = 1;
        feat_num = units.n_cols - 1;
    }
    this->p_treatments = std::make_shared<RMat>(data.cols(treat_ist, treat_ist));
    this->p_data = std::make_shared<RMat>(data.cols(feat_ist, data.n_cols-1));
    this->p_targets = std::make_shared<RMat>(series);

    /**/
    this->n_instances = units.n_rows;
    this->n_features = feat_num;
    this->n_targets = 1;
    this->n_treatments = n_treatments;
    this->n_treated = n_treated;
    this->n_periods = (series.n_cols-feat_ist-n_treatments)/2;

    // this->head_targets = arma::field<std::tuple<std::string, size_t>>(1, n_periods);
}

void didtree::Dataset::descript()const{
    BOOST_LOG_SEV(didtree::lg, didtree::INFO)<<"*************dataset*******************";
    BOOST_LOG_SEV(didtree::lg, didtree::INFO)<<"#inst,#feat:\t"<<n_instances<<","<<n_features;
    BOOST_LOG_SEV(didtree::lg, didtree::INFO)<<"#treatment:\t"<<n_treatments<<"#treated"<<n_treated;
    BOOST_LOG_SEV(didtree::lg, didtree::INFO)<<"#period:\t"<<n_periods;
    BOOST_LOG_SEV(didtree::lg, didtree::INFO)<<"***************************************";
}