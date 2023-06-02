/** @file trigonometric_example.cpp

    @brief Usage of preAApp to solve a trigonometric problem

    This Source Code Form is subject to the terms of the GNU Affero General
    Public License v3.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at https://www.gnu.org/licenses/agpl-3.0.en.html.

    Author(s): Ye Ji <jiyess@outlook.com>
*/

//! [Include namespace]
#include "preAA.h"
#include "iostream"

using namespace preAApp;
//! [Include namespace]

// Trigonometric nonlinear equations, Mor ́e et al. (1981)
Residual_t residual = [](const VectorX &u) {
  auto n = u.size();
  VectorX F(n);

  double fixedPt = M_PI/4.0;

  double sum = 0.0, sumFixedPt = 0.0;
  for (auto i=0; i!=n; ++i) {
    sum += cos(u(i));
    sumFixedPt += cos(fixedPt);
  }

  for (auto i=0; i!=n; ++i) {
    F(i) = -sum-i*cos(u(i))-sin(u(i)) +
        sumFixedPt+i*cos(fixedPt)+sin(fixedPt);
  }

  return F;
};

Jacobian_t jacobian = [](VectorX const &u) {
  auto n = u.size();
  ColMajorSparseMatrix jac(n, n);

  for (auto i = 0; i!=n; ++i){
    double val = (i+1.0) * sin(u(i)) - cos(u(i));
    for (auto j = 0; j!=n; ++j) {
     if (i == j) {
       jac.coeffRef(i,j) = val;
     }
     else {
       jac.coeffRef(i,j) = sin(u(j));
     }
    }
  }
  jac.makeCompressed();

  return jac;
};

int main(int argc, char *argv[]) {
  int m = 3, n = 500;
  if (argc == 2) {m = std::stoi(argv[1]);}

  preAAParam<> param;
  param.m = m;
  param.usePreconditioning = true;
  param.updatePreconditionerStep = 10;
  param.epsilon = 1e-10;

  AndersonAcceleration<> AASolver(param);

  VectorX initialGuess = VectorX::Ones(n);
  std::cout << n << "\n";

  measureTime([&]() {
    VectorX sol = AASolver.compute(initialGuess, residual, jacobian);
  });
}

