#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

std::vector<std::vector<std::vector<float> > > compute(const std::vector<std::vector<float> >& pts_s2_float, const std::vector<float>& pts_norm, int r, int b, const std::vector<float>& weights) {
    float interval = 1.f / r;
    std::vector<std::vector<std::vector<std::vector<std::vector<float> > > > > grids;
    std::vector<std::vector<std::vector<float> > > features;

    grids.resize(r);
    features.resize(r);
    for (auto &beta: grids) {
        beta.resize(2 * b);
        for (auto &alpha: beta) {
            alpha.resize(2 * b);
        }
    }
    for (auto &beta: features) {
        beta.resize(2 * b);
        for (auto &alpha: beta) {
            alpha.resize(2 * b, 0);
        }
    }

    for (size_t i = 0; i < pts_s2_float.size(); i++) {
        int r_idx = int(pts_norm[i] / interval);
        if (r_idx > r - 1) r_idx = r - 1;
        int beta_idx = int(pts_s2_float[i][0] + 0.5f);
        if (beta_idx > 2 * b - 1) beta_idx = 2 * b - 1;
        int alpha_idx = int(pts_s2_float[i][1] + 0.5f);
        if (alpha_idx > 2 * b - 1) alpha_idx = 2 * b - 1;

        // std::cout << r_idx << ", " << pts_norm[i] << std::endl;
        grids[r_idx][beta_idx][alpha_idx].emplace_back(std::vector<float>{pts_norm[i], pts_s2_float[i][0], pts_s2_float[i][1]});
    }

    for (size_t i = 0; i < r; i++) {
        for (size_t j = 0; j < 2 * b; j++) {
            for (size_t k = 0; k < 2 * b; k++) {
                float left = std::max(0.f, k - 0.5f / weights[j]);
                float right = std::min(2.f * b, k + 0.5f / weights[j]);
                float sum = 0.f;
                int cnt = 0;
                for (int m = int(left + 0.5f); m < int(right + 0.5f); m++) {
                    for (int n = 0; n < grids[i][j][m].size(); n++) {
                        if (grids[i][j][m][n][2] > left && grids[i][j][m][n][2] < right) {
                            sum += 1.f - std::abs(grids[i][j][m][n][0] / interval - (i + 1));
                            cnt++;
                        }
                    }
                    if (i < r - 1) {
                        for (int n = 0; n < grids[i + 1][j][m].size(); n++) {
                            if (grids[i + 1][j][m][n][2] > left && grids[i + 1][j][m][n][2] < right) {
                                sum += 1.f - std::abs(grids[i + 1][j][m][n][0] / interval - (i + 1));
                                cnt++;
                            }
                        }
                    }
                }
                if (cnt > 0) {
                    features[i][j][k] = sum / cnt;
//                    std::cout << features[i][j][k] << std::endl;
                }
            }
        }
    }
    return features;
}

PYBIND11_MODULE(sampling, m) {
    m.def("compute", &compute, py::arg("pts_s2_float"), py::arg("pts_norm"), py::arg("r"), py::arg("b"), py::arg("weights"));
}