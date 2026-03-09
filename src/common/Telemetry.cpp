#include "Telemetry.h"

using namespace ecv;

std::shared_ptr<mini_server::IpcServer> Telemetry::server;
Telemetry::LogLevel Telemetry::logLevel;