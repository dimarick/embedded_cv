#include "Telemetry.h"

using namespace ecv;

std::shared_ptr<mini_server::BroadcastingServer> Telemetry::server;
Telemetry::LogLevel Telemetry::logLevel;