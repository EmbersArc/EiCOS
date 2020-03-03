#pragma once

#include <iostream>

void mu_assert(const char *message, bool f)
{
    if (not f)
    {
        std::cout << message << std::endl;
        std::abort();
    }
}

extern int tests_run;

template <typename fun>
char *mu_run_test(fun test)
{
    char *message = test();
    tests_run++;
    if (message)
        return message;
    return 0;
}