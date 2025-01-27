#pragma once
#include <cmath>
#include <vector>
#include <array>
#include <iostream>

template <typename T>
struct Vector3d
{
    __device__ __host__ T norm2() const
    {
        T dist2 = x * x +
                  y * y +
                  z * z;
        return dist2;
    }

    __device__ __host__ T norm() const
    {
        return sqrt(norm2());
    }

    __device__ __host__ Vector3d<T> scale(T factor) const
    {
        Vector3d res = *this;
        res.x *= factor;
        res.y *= factor;
        res.z *= factor;
        return res;
    }
    __device__ __host__ T dot(const Vector3d<T> &other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }
    __device__ __host__ Vector3d<T> cross(const Vector3d<T> &other) const
    {
        Vector3d res;
        res.x = y * other.z - z * other.y;
        res.y = z * other.x - x * other.z;
        res.z = x * other.y - y * other.x;
        return res;
    }
    __device__ __host__ Vector3d<T> operator+(const Vector3d &other) const
    {
        Vector3d res = *this;
        res.x += other.x;
        res.y += other.y;
        res.z += other.z;
        return res;
    }
    __device__ __host__ Vector3d<T> operator-(const Vector3d &other) const
    {
        Vector3d res = *this;
        res.x -= other.x;
        res.y -= other.y;
        res.z -= other.z;
        return res;
    }
    __device__ __host__ Vector3d<T> operator+=(const Vector3d &other)
    {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
    __host__ friend std::ostream &operator<<(std::ostream &o, const Vector3d &self)
    {
        return o << self.x << ' ' << self.y << ' ' << self.z;
    }
    template <typename U>
    __device__ __host__ Vector3d<U> to() const
    {
        Vector3d<U> res;
        res.x = static_cast<U>(x);
        res.y = static_cast<U>(y);
        res.z = static_cast<U>(z);
        return res;
    }
    T x;
    T y;
    T z;
};

enum struct Axis
{
    x,
    y,
    z,
    empty
};

template <typename T>
class Rotation
{
public:
    Rotation() : rotate_matrix{0}
    {
    }
    __device__ __host__ Rotation(Axis axis, T angle)
    {
        switch (axis)
        {
        case Axis::x:
            rotate_matrix[0][0] = 1.0;
            rotate_matrix[0][1] = 0.0;
            rotate_matrix[0][2] = 0.0;
            rotate_matrix[1][0] = 0.0;
            rotate_matrix[1][1] = cos(angle);
            rotate_matrix[1][2] = -sin(angle);
            rotate_matrix[2][0] = 0.0;
            rotate_matrix[2][1] = sin(angle);
            rotate_matrix[2][2] = cos(angle);
            break;
        case Axis::y:
            rotate_matrix[0][0] = cos(angle);
            rotate_matrix[0][1] = 0.0;
            rotate_matrix[0][2] = -sin(angle);
            rotate_matrix[1][0] = 0.0;
            rotate_matrix[1][1] = 1.0;
            rotate_matrix[1][2] = 0.0;
            rotate_matrix[2][0] = sin(angle);
            rotate_matrix[2][1] = 0.0;
            rotate_matrix[2][2] = cos(angle);
            break;
        case Axis::z:
            rotate_matrix[0][0] = cos(angle);
            rotate_matrix[0][1] = -sin(angle);
            rotate_matrix[0][2] = 0.0;
            rotate_matrix[1][0] = sin(angle);
            rotate_matrix[1][1] = cos(angle);
            rotate_matrix[1][2] = 0.0;
            rotate_matrix[2][0] = 0.0;
            rotate_matrix[2][1] = 0.0;
            rotate_matrix[2][2] = 1.0;
            break;
        case Axis::empty:
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    rotate_matrix[i][j] = angle;
                }
            }
            break;
        }
    }
    template <typename U>
    __device__ __host__ Vector3d<U> rotate(const Vector3d<U> &vec3d) const
    {
        Vector3d<U> res{0, 0, 0};
        res.x += rotate_matrix[0][0] * vec3d.x;
        res.x += rotate_matrix[0][1] * vec3d.y;
        res.x += rotate_matrix[0][2] * vec3d.z;

        res.y += rotate_matrix[1][0] * vec3d.x;
        res.y += rotate_matrix[1][1] * vec3d.y;
        res.y += rotate_matrix[1][2] * vec3d.z;

        res.z += rotate_matrix[2][0] * vec3d.x;
        res.z += rotate_matrix[2][1] * vec3d.y;
        res.z += rotate_matrix[2][2] * vec3d.z;
        return res;
    }

    __device__ __host__ Rotation compose(const Rotation &other) const
    {
        Rotation res(Axis::empty, 0);
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                for (int k = 0; k < 3; ++k)
                {
                    res.rotate_matrix[i][j] += rotate_matrix[i][k] *
                                               other.rotate_matrix[k][j];
                }
            }
        }
        return res;
    }
    __host__ friend std::ostream &operator<<(std::ostream &o, const Rotation &self)
    {
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                o << self.rotate_matrix[i][j] << ' ';
            }
            o << '\n';
        }
        return o;
    }

    template <typename U>
    friend class Rotation;

    template <typename U>
    Rotation<U> to() const
    {
        Rotation<U> res;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                res.rotate_matrix[i][j] = static_cast<U>(rotate_matrix[i][j]);
            }
        }
        return res;
    }

private:
    std::array<std::array<T, 3>, 3> rotate_matrix;
};

template <typename T>
Rotation(Axis axis, T angle) -> Rotation<T>;