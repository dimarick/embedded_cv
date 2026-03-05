#ifndef EMBEDDED_CV_TELEMETRY_H
#define EMBEDDED_CV_TELEMETRY_H

#include <string>
#include <chrono>
#include <any>
#include <iostream>
#include <BroadcastingServer.h>

namespace ecv {
    class Telemetry {
    public:
        enum LogLevel {
            DEBUG = 0,
            INFO = 1,
            WARN = 2,
            ERROR = 3,
        };
    private:
        static double now() {
            return std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        }
        static const char *level(LogLevel l) {
            switch (l) {
                case LogLevel::DEBUG:
                    return "debug";
                case LogLevel::INFO:
                    return "info";
                case LogLevel::WARN:
                    return "warn";
                case LogLevel::ERROR:
                    return "error";
            };
            return "";
        }

        static std::shared_ptr<mini_server::BroadcastingServer> server;
        static LogLevel logLevel;

        static std::string datetimeFromFloat(double ts) {
            using namespace std::chrono;
            auto sec = duration_cast<seconds>(duration<double>(ts));
            auto ms = duration_cast<milliseconds>(duration<double>(ts)) - sec;
            sys_seconds tp{sec};
            return std::format("{:%Y-%m-%d %H:%M:%S}.{:03}", tp, ms.count());
        }

        static void _status(const std::string &component, const std::vector<std::string> &properties, const std::vector<std::string> &quotedStrings) {
            std::ostringstream str;
            if (properties.size() != quotedStrings.size()) {
                throw std::runtime_error("properties and values should have equal size");
            }

            str << "[" << std::quoted("STATUS") << "," << std::quoted(component);
            for (int i = 0; i < properties.size(); ++i) {
                str << "," << std::quoted(properties[i]) << "," << quotedStrings[i];
            }
            str << "]";

            std::cout << str.str() << std::endl;
            server->broadcast(str.str());
        }
    public:
        static void setServer(std::shared_ptr<mini_server::BroadcastingServer> s) {
            server = std::move(s);
        }
        static void setLogLevel(LogLevel l) {
            logLevel = l;
        }
        static void log(LogLevel l, const std::string &message) {
            if (l < logLevel) {
                return;
            }
            std::ostringstream str;
            str << "[" << std::quoted("LOG") << "," << std::quoted(level(l)) << "," << now() << "," << std::quoted(message) << "]";
            if (l != ERROR) {
                std::cout << ":[" << datetimeFromFloat(now()) << "]\t" << level(l) << "\t" << message << std::endl;;
            } else {
                std::cerr << ":[" << datetimeFromFloat(now()) << "]\t" << level(l) << "\t" << message << std::endl;;
            }
            server->broadcast(str.str());
        }
        static void debug(const std::string &message) {
            log(DEBUG, message);
        }
        static void info(const std::string &message) {
            log(INFO, message);
        }
        static void warn(const std::string &message) {
            log(WARN, message);
        }
        static void error(const std::string &message) {
            log(ERROR, message);
        }
        static void status(const std::string &component, const std::string &property, const std::string &string) {
            std::ostringstream s;
            s << std::quoted(string);
            _status(component, {property}, {s.str()});
        }

        template<typename T> static void status(const std::string &component, const std::string &property, T value) {
            _status(component, {property}, {std::to_string(value)});
        }
        static void status(const std::string &component, const std::vector<std::string> &properties, const std::vector<std::string> &strings) {
            std::vector<std::string> quoted(strings.size());

            for (int i = 0; i < strings.size(); ++i) {
                std::ostringstream s;
                s << std::quoted(strings[i]);
                quoted[i] = s.str();
            }

            _status(component, properties, quoted);
        }

        template<typename T> static void status(const std::string &component, const std::vector<std::string> &properties, const std::vector<T> &values) {
            std::vector<std::string> str(values.size());
            for (int i = 0; i < values.size(); ++i) {
                str[i] = std::to_string(values[i]);
            }
            _status(component, properties, str);
        }
    };
}

#endif //EMBEDDED_CV_TELEMETRY_H
