# include <armadillo>
# include <iostream>

# include "dataset.h"
// # include "log.h"


didtree::Dataset::Dataset(const std::string&path){
    field<std::string> hdata, htarget, htreatment;
    auto units = RMat();
    auto series = RMat();
    units.load(csv_name(path+"\\features.csv", hdata));
    series.load(csv_name(path+"\\targets.csv", htarget));
    std::cout<<hdata.size()<<std::endl;
}

