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

#include <set>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

namespace visnav {

void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double cam_z_threshold,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {
  projected_points.clear();
  projected_track_ids.clear();

  for (auto lmark : landmarks) {
    Eigen::Vector3d p_3d = current_pose.inverse() * lmark.second.p;
    if (p_3d[2] < cam_z_threshold) continue;

    Eigen::Vector2d p_2d = cam->project(p_3d);
    if (p_2d[0] < 0 || p_2d[0] >= 752 || p_2d[1] < 0 || p_2d[1] >= 480)
      continue;
    projected_points.push_back(p_2d);
    projected_track_ids.push_back(lmark.first);
  }
}

void find_matches_landmarks(
    const KeypointsData& kdl, const Landmarks& landmarks,
    const Corners& feature_corners,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    const std::vector<TrackId>& projected_track_ids,
    const double match_max_dist_2d, const int feature_match_max_dist,
    const double feature_match_test_next_best, MatchData& md) {
  md.matches.clear();

  for (size_t i = 0; i < kdl.corners.size(); i++) {
    int best_idx = -1;
    size_t best2_dist = 500, best_dist = 500;

    for (size_t j = 0; j < projected_points.size(); j++) {
      if ((kdl.corners[i] - projected_points[j]).norm() > match_max_dist_2d)
        continue;

      size_t lm_dist = 500;
      for (auto feature : landmarks.at(projected_track_ids[j]).obs) {
        auto des = feature_corners.at(feature.first)
                       .corner_descriptors.at(feature.second);
        if ((kdl.corner_descriptors.at(i) ^ des).count() < lm_dist)
          lm_dist = (kdl.corner_descriptors.at(i) ^ des).count();
      }

      if (lm_dist <= best_dist) {
        best2_dist = best_dist;

        best_dist = lm_dist;
        best_idx = projected_track_ids[j];
      } else if (lm_dist < best2_dist) {
        best2_dist = lm_dist;
      }
    }

    if (best_dist < feature_match_max_dist &&
        best_dist * feature_match_test_next_best <= best2_dist) {
      md.matches.push_back(std::make_pair(i, best_idx));
    }
  }
}

void localize_camera(const std::shared_ptr<AbstractCamera<double>>& cam,
                     const KeypointsData& kdl, const Landmarks& landmarks,
                     const double reprojection_error_pnp_inlier_threshold_pixel,
                     const MatchData& md, Sophus::SE3d& T_w_c,
                     std::vector<int>& inliers) {
  inliers.clear();

  if (md.matches.size() == 0) {
    T_w_c = Sophus::SE3d();
    return;
  }

  double flen = (cam->data()[0] + cam->data()[1]) / 2;
  double ransac_thresh =
      1.0 -
      cos(atan(reprojection_error_pnp_inlier_threshold_pixel * 0.5 / flen));

  opengv::bearingVectors_t bvs;
  opengv::points_t pts;
  for (auto match : md.matches) {
    bvs.push_back(cam->unproject(kdl.corners[match.first]));
    pts.push_back(landmarks.at(match.second).p);
  }

  using namespace opengv::sac_problems::absolute_pose;

  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bvs, pts);

  std::shared_ptr<AbsolutePoseSacProblem> absposeproblem_ptr(
      new AbsolutePoseSacProblem(adapter, AbsolutePoseSacProblem::KNEIP));

  opengv::sac::Ransac<AbsolutePoseSacProblem> ransac;

  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ = ransac_thresh;
  ransac.computeModel();

  opengv::transformation_t tform;
  absposeproblem_ptr->optimizeModelCoefficients(
      ransac.inliers_, ransac.model_coefficients_, tform);

  absposeproblem_ptr->selectWithinDistance(tform, ransac_thresh, inliers);

  T_w_c = Sophus::SE3d(tform.block<3, 3>(0, 0), tform.block<3, 1>(0, 3));
}

void add_new_landmarks(const TimeCamId tcidl, const TimeCamId tcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Sophus::SE3d& T_w_c0, const Calibration& calib_cam,
                       const std::vector<int> inliers,
                       const MatchData& md_stereo, const MatchData& md,
                       Landmarks& landmarks, TrackId& next_landmark_id) {
  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  std::set<std::pair<FeatureId, FeatureId>> added;
  for (auto i : inliers) {
    landmarks[md.matches[i].second].obs.emplace(tcidl, md.matches[i].first);

    for (auto st_in : md_stereo.inliers)
      if (st_in.first == md.matches[i].first) {
        added.emplace(st_in);
        landmarks[md.matches[i].second].obs.emplace(tcidr, st_in.second);
        break;
      }
  }

  opengv::bearingVectors_t bvs1, bvs2;
  for (auto match : md_stereo.inliers)
    if (added.count(match) == 0) {
      bvs1.push_back(calib_cam.intrinsics[tcidl.second]->unproject(
          kdl.corners[match.first]));
      bvs2.push_back(calib_cam.intrinsics[tcidr.second]->unproject(
          kdr.corners[match.second]));
    }

  opengv::relative_pose::CentralRelativeAdapter adapter(bvs1, bvs2, t_0_1,
                                                        R_0_1);
  int idx = 0;
  for (auto match : md_stereo.inliers)
    if (added.count(match) == 0) {
      Landmark lmark;

      lmark.p = T_w_c0 * opengv::triangulation::triangulate(adapter, idx++);
      lmark.obs.emplace(tcidl, match.first);
      lmark.obs.emplace(tcidr, match.second);

      landmarks[next_landmark_id++] = lmark;
    }
}

void remove_old_keyframes(const TimeCamId tcidl, const int max_num_kfs,
                          Cameras& cameras, Landmarks& landmarks,
                          Landmarks& old_landmarks,
                          std::set<FrameId>& kf_frames) {
  kf_frames.emplace(tcidl.first);

  std::set<FrameId> pairs;
  for (auto kv : cameras)
    if (cameras.count(std::make_pair(kv.first.first, 1 - kv.first.second)) > 0)
      pairs.emplace(kv.first.first);

  if (pairs.size() <= max_num_kfs) return;

  while (kf_frames.size() > max_num_kfs) {
    for (auto cam : cameras) {
      if (cam.first.first == *(kf_frames.begin())) {
        auto fid = cam.first.first;
        cameras.erase(std::make_pair(fid, 0));
        cameras.erase(std::make_pair(fid, 1));

        for (auto& lmark : landmarks) {
          lmark.second.obs.erase(std::make_pair(fid, 0));
          lmark.second.obs.erase(std::make_pair(fid, 1));
          if (lmark.second.obs.empty()) old_landmarks.emplace(lmark);
        }
      }
    }
    kf_frames.erase(kf_frames.begin());
  }

  for (auto lmark : old_landmarks) landmarks.erase(lmark.first);
}
}  // namespace visnav
