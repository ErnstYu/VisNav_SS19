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

#include <bitset>
#include <set>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include <visnav/camera_models.h>
#include <visnav/common_types.h>

namespace visnav {

void computeEssential(const Sophus::SE3d& T_0_1, Eigen::Matrix3d& E) {
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  Eigen::Matrix3d S;
  const Eigen::Vector3d t_n = t_0_1.normalized();

  S << 0, -t_n(2), t_n(1), t_n(2), 0, -t_n(0), -t_n(1), t_n(0), 0;

  E = S * R_0_1;
}

void findInliersEssential(const KeypointsData& kd1, const KeypointsData& kd2,
                          const std::shared_ptr<AbstractCamera<double>>& cam1,
                          const std::shared_ptr<AbstractCamera<double>>& cam2,
                          const Eigen::Matrix3d& E,
                          double epipolar_error_threshold, MatchData& md) {
  md.inliers.clear();

  for (size_t j = 0; j < md.matches.size(); j++) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];

    const Eigen::Vector3d p0_3d = cam1->unproject(p0_2d);
    const Eigen::Vector3d p1_3d = cam2->unproject(p1_2d);

    if (abs(p0_3d.transpose() * E * p1_3d) < epipolar_error_threshold)
      md.inliers.push_back(md.matches[j]);
  }
}

void findInliersRansac(const KeypointsData& kd1, const KeypointsData& kd2,
                       const std::shared_ptr<AbstractCamera<double>>& cam1,
                       const std::shared_ptr<AbstractCamera<double>>& cam2,
                       const double ransac_thresh, const int ransac_min_inliers,
                       MatchData& md) {
  md.inliers.clear();

  opengv::bearingVectors_t bvs1, bvs2;

  for (auto match : md.matches) {
    bvs1.push_back(cam1->unproject(kd1.corners[match.first]));
    bvs2.push_back(cam2->unproject(kd2.corners[match.second]));
  }

  using namespace opengv::sac_problems::relative_pose;

  opengv::relative_pose::CentralRelativeAdapter adapter(bvs1, bvs2);

  std::shared_ptr<CentralRelativePoseSacProblem> relposeproblem_ptr(
      new CentralRelativePoseSacProblem(adapter,
                                        CentralRelativePoseSacProblem::NISTER));

  opengv::sac::Ransac<CentralRelativePoseSacProblem> ransac;

  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ransac_thresh;
  ransac.computeModel();

  auto tform = ransac.model_coefficients_;
  relposeproblem_ptr->optimizeModelCoefficients(
      ransac.inliers_, ransac.model_coefficients_, tform);

  auto inliers_ = ransac.inliers_;
  relposeproblem_ptr->selectWithinDistance(tform, ransac_thresh, inliers_);

  md.T_i_j = Sophus::SE3d(tform.block<3, 3>(0, 0), tform.block<3, 1>(0, 3));

  if (int(inliers_.size()) < ransac_min_inliers) return;
  for (auto idx : inliers_) md.inliers.push_back(md.matches[idx]);
}
}  // namespace visnav
