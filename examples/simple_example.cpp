/** @file simple_example.cpp

    @brief Tutorial on how to use preAApp by solving a simple problem

    This Source Code Form is subject to the terms of the GNU Affero General
    Public License v3.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at https://www.gnu.org/licenses/agpl-3.0.en.html.

    Author(s): Ye Ji <jiyess@outlook.com>
*/

//! [Include namespace]
#include "preAA.h"

using namespace preAApp;
//! [Include namespace]

Residual_t residual = [](VectorX const &u) {
  VectorX F(2);

  F(0) = cos(0.5 * (u(0)+u(1)));
  F(1) = F(0) + 1e-8 * sin(u(1)*u(1));
  return F;
};

Jacobian_t jacobian = [](VectorX const &u) {
  ColMajorSparseMatrix jac(2, 2);

  double val = -0.5*sin(0.5*(u(0)+u(1)));
  jac.coeffRef(0,0) = val;
  jac.coeffRef(1,0) = val;
  jac.coeffRef(0,1) = val;
  jac.coeffRef(1,1) = val + 2e-8 * u(1) * cos(pow(u(1), 2));
  jac.makeCompressed();

  return jac;
};

int main(int argc, char *argv[]) {
  int m = 2;
  if (argc == 2) {m = std::stoi(argv[1]);}

  preAAParam<> param;
  param.m = m;
  param.usePreconditioning = true;
  param.updatePreconditionerStep = 5;
  param.epsilon = 1e-10;

  AndersonAcceleration<> AASolver(param);

  VectorX initialGuess = VectorX::Ones(2);

  measureTime([&]() {
    VectorX sol = AASolver.compute(initialGuess, residual, jacobian);
  });
}