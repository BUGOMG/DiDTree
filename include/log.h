#ifndef AUDIO_RECO_MODULES_COMMON_BOOSTLOGGING_H_
#define AUDIO_RECO_MODULES_COMMON_BOOSTLOGGING_H_

#include <cstddef>
#include <string>
#include <ostream>
#include <fstream>
#include <iomanip>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared_object.hpp>
#include <boost/phoenix/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/sources/basic_logger.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/severity_channel_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/attributes/scoped_attribute.hpp>
#include <boost/log/utility/value_ref.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/trivial.hpp>

namespace logging = boost::log;
namespace src = boost::log::sources;
namespace expr = boost::log::expressions;
namespace sinks = boost::log::sinks;
namespace attrs = boost::log::attributes;
namespace keywords = boost::log::keywords;
namespace didtree
{

  enum severity_level {
    TRACE,
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    FATAL,
    NODE
  };

  BOOST_LOG_ATTRIBUTE_KEYWORD(line_id, "LineID", unsigned int)
  BOOST_LOG_ATTRIBUTE_KEYWORD(severity, "Severity", severity_level)
  BOOST_LOG_ATTRIBUTE_KEYWORD(tag_attr, "Tag", std::string)

  std::ostream& operator<<(std::ostream& strm, severity_level level);
  void InitBoostLog(severity_level logging_level, std::string module_name);

} // namespace didtree
#endif //AUDIO_RECO_MODULES_COMMON_BOOSTLOGGING_H_