#pragma once
#include <chrono>
#include <stack>
#include <iostream>

namespace cudaSolarTomography
{
    namespace stdc = std::chrono;

    class Timer
    {
    public:
        void start()
        {
            start_points.push(stdc::high_resolution_clock::now());
        }

        void stop(std::string_view message)
        {
            if (!start_points.empty())
            {
                std::cerr << message << " " << stdc::duration_cast<stdc::milliseconds>(stdc::high_resolution_clock::now() - start_points.top()).count() << "ms\n";
                start_points.pop();
            }
            else
            {
                std::cerr << "no pairing start point!\n";
                std::terminate();
            }
        }

    private:
        std::stack<stdc::high_resolution_clock::time_point> start_points;
    };
}
