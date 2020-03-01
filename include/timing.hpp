#pragma once

#include <chrono>

double tic()
{
    const std::chrono::duration<double, std::milli> s = std::chrono::system_clock::now().time_since_epoch();

    return s.count();
}

double toc(double start)
{
    return tic() - start;
}
