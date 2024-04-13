#include <cpptrace/cpptrace.hpp>

int main(int argc, const char* argv[]) {
    cpptrace::register_terminate_handler();
    return 0;
}
