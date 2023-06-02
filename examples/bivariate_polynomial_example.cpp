/** @file bivariate_polynomial_example.cpp

    @brief Usage of preAApp to solve a bivariate polynomial problem

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

// 2D nonlinear polynomial systems, Kelley and Suresh (1983)
Residual_t residual = [](const VectorX &u) {
  VectorX F(2);

  double epsilon = 1e-8;
  F(0) = u(0) - 1 + pow(u(1) - 3.0, 2.0);
  F(1) = epsilon * (u(1) - 3.0) + 1.5 * (u(0) - 1) * (u(1) - 3.0) +
      pow(u(1) - 3, 2.0) + pow(u(1) - 3, 3.0);

  return F;
};

Jacobian_t jacobian = [](VectorX const &u) {
  ColMajorSparseMatrix jac(2, 2);

  double epsilon = 1e-8;
  jac.coeffRef(0, 0) = 1.0;
  jac.coeffRef(1, 0) = u(0) - 1 + 2.0 * (u(1) - 3.0);
  jac.coeffRef(0, 1) = 1.5 * (u(1) - 3.0);
  jac.coeffRef(1, 1) = epsilon + 1.5 * (u(0) - 1.0) + 2.0 * (u(1) - 3.0) +
      3.0 * pow(u(1)-3.0, 2.0);
  jac.makeCompressed();

  return jac;
};

int main(int argc, char *argv[]) {
  int m = 2;
  if (argc == 2) { m = std::stoi(argv[1]); }

  preAAParam<> param;
  param.m = m;
  param.usePreconditioning = true;
  param.updatePreconditionerStep = 1;
  param.epsilon = 1e-10;

  AndersonAcceleration<> AASolver(param);

  VectorX initialGuess = VectorX::Ones(2);

  measureTime([&]() {
    VectorX sol = AASolver.compute(initialGuess, residual, jacobian);
  });
}

