/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <sophus/se3.hpp>

//#include <visnav/common_types.h>

namespace visnav {

// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(
    const Eigen::Matrix<T, 3, 1>& xi) {
  Eigen::Matrix<T, 3, 3> ans;
  Eigen::Matrix<T, 3, 1> a = xi.normalized();
  T theta = xi.norm();

  if (theta == 0) return Eigen::Matrix<T, 3, 3>::Identity();

  Eigen::Matrix<T, 3, 3> ssa;
  ssa << 0, -a(2, 0), a(1, 0), a(2, 0), 0, -a(0, 0), -a(1, 0), a(0, 0), 0;

  ans = cos(theta) * Eigen::Matrix<T, 3, 3>::Identity() +
        (1 - cos(theta)) * a * a.transpose() + sin(theta) * ssa;

  return ans;
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {
  T norm = acos((mat.trace() - 1) / 2);

  if (norm == 0) return Eigen::Matrix<T, 3, 1>::Zero();

  Eigen::Matrix<T, 3, 1> a(mat(2, 1) - mat(1, 2), mat(0, 2) - mat(2, 0),
                           mat(1, 0) - mat(0, 1));
  a = norm * 1 / (2 * sin(norm)) * a;

  return a;
}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(
    const Eigen::Matrix<T, 6, 1>& xi) {
  Eigen::Matrix<T, 4, 4> ans;
  Eigen::Matrix<T, 3, 1> rho = xi.block(0, 0, 3, 1);
  Eigen::Matrix<T, 3, 1> phi = xi.block(3, 0, 3, 1);
  Eigen::Matrix<T, 3, 1> a = phi.normalized();
  T theta = phi.norm();

  if (theta == 0) {
    ans.block(0, 0, 3, 3) = Eigen::Matrix<T, 3, 3>::Identity();
    ans.block(0, 3, 3, 1) = Eigen::Matrix<T, 3, 1>::Zero();
    ans.block(3, 0, 1, 3) = Eigen::Matrix<T, 1, 3>::Zero();
    ans(3, 3) = 1;

    return ans;
  }

  Eigen::Matrix<T, 3, 3> ssa;
  ssa << 0, -a(2, 0), a(1, 0), a(2, 0), 0, -a(0, 0), -a(1, 0), a(0, 0), 0;

  Eigen::Matrix<T, 3, 3> R, J;
  R = cos(theta) * Eigen::Matrix<T, 3, 3>::Identity() +
      (1 - cos(theta)) * a * a.transpose() + sin(theta) * ssa;
  J = sin(theta) / theta * Eigen::Matrix<T, 3, 3>::Identity() +
      (1 - sin(theta) / theta) * a * a.transpose() +
      (1 - cos(theta)) / theta * ssa;

  ans.block(0, 0, 3, 3) = R;
  ans.block(0, 3, 3, 1) = J * rho;
  ans.block(3, 0, 1, 3) = Eigen::Matrix<T, 1, 3>::Zero();
  ans(3, 3) = 1;

  return ans;
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {
  Eigen::Matrix<T, 3, 3> R = mat.block(0, 0, 3, 3);
  Eigen::Matrix<T, 3, 1> t = mat.block(0, 3, 3, 1);

  T norm = acos((R.trace() - 1) / 2);

  if (norm == 0) return Eigen::Matrix<T, 6, 1>::Zero();

  Eigen::Matrix<T, 3, 1> phi(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0),
                             R(1, 0) - R(0, 1));
  phi = norm * 1 / (2 * sin(norm)) * phi;

  Eigen::Matrix<T, 3, 1> rho;
  Eigen::Matrix<T, 3, 1> a = phi.normalized();
  T theta = phi.norm();

  Eigen::Matrix<T, 3, 3> ssa;
  ssa << 0, -a(2, 0), a(1, 0), a(2, 0), 0, -a(0, 0), -a(1, 0), a(0, 0), 0;

  Eigen::Matrix<T, 3, 3> J;
  J = sin(theta) / theta * Eigen::Matrix<T, 3, 3>::Identity() +
      (1 - sin(theta) / theta) * a * a.transpose() +
      (1 - cos(theta)) / theta * ssa;

  rho = J.inverse() * t;

  Eigen::Matrix<T, 6, 1> ans;
  ans.block(0, 0, 3, 1) = rho;
  ans.block(3, 0, 3, 1) = phi;

  return ans;
}

}  // namespace visnav
