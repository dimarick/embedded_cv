#include "Telemetry.h"

using namespace ecv;

std::shared_ptr<mini_server::IpcServer> Telemetry::server;
Telemetry::LogLevel Telemetry::logLevel;
std::unordered_map<std::string, std::string> Telemetry::valueCache;