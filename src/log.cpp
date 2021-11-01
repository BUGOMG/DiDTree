#include "log.h"

namespace didtree
{

  std::ostream &operator<<(std::ostream &strm, severity_level level)
  {
    static const char *strings[] = {
        "trace",
        "debug",
        "info",
        "warning",
        "error",
        "fatal",
        "node"};

    if (static_cast<std::size_t>(level) < sizeof(strings) / sizeof(*strings))
      strm << strings[level];
    else
      strm << static_cast<int>(level);

    return strm;
  }

  void InitBoostLog(severity_level logging_level, std::string module_name)
  {
    // Setup the common formatter for all sinks
    logging::formatter fmt = expr::stream
                             << std::setw(3) << std::setfill(' ') << line_id << std::setfill(' ')
                             << ": [" << severity << "]\t"
                             << expr::if_(expr::has_attr(tag_attr))
                                    [expr::stream << "[" << tag_attr << "] "]
                             << expr::smessage;

    // Initialize sinks
    boost::shared_ptr<logging::core> core = logging::core::get();
    typedef sinks::synchronous_sink<sinks::text_ostream_backend> text_sink;
    boost::shared_ptr<text_sink> sink = boost::make_shared<text_sink>();

    sink->locked_backend()->add_stream(
        boost::make_shared<std::ofstream>(module_name + ".log"));
    sink->set_formatter(fmt);
    sink->set_filter(severity != NODE && severity >= logging_level);
    core->add_sink(sink);

    sink = boost::make_shared<text_sink>();
    sink->locked_backend()->add_stream(
        boost::make_shared<std::ofstream>(module_name + "-node.log"));
    sink->set_filter(severity == NODE);
    core->add_sink(sink);

    // Add attributes
    logging::add_common_attributes();
  }
  
  src::severity_logger<severity_level> lg;
} // namespace didtree
