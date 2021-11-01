#include <iostream>
#include <vector>
#include <string>
#include <stack>
#include <limits>
#include <algorithm>

#include "dataset.h"
#include "log.h"

using namespace std;
// extern src::severity_logger<severity_level> didtree::lg;
int main(int argc, char **argv)
{
    didtree::InitBoostLog(didtree::INFO, "DiDTree");
    BOOST_LOG_SEV(didtree::lg, didtree::INFO)<<"info"<<"\tok";
    didtree::Dataset dataset("D:\\work\\DiDTree\\data\\binary_small_0.93");
    dataset.descript();
}