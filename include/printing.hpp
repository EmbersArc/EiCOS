constexpr bool debug_printing = false;

#if __has_include(<fmt/format.h>)

#include <fmt/format.h>
#include <fmt/ostream.h>

using fmt::format;
using fmt::print;

template <typename... Params>
void print_dbg(Params &&... params)
{
    if constexpr (debug_printing)
    {
        fmt::print(std::forward<Params>(params)...);
    }
}

#else

template <typename... Params>
void print_dbg(Params &&... params)
{
}
template <typename... Params>
void print(Params &&... params)
{
}
template <typename... Params>
std::string format(Params &&... params)
{
    return "";
}

#endif