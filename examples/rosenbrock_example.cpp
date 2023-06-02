/** @file rosenbrock_example.cpp

    @brief Tutorial on how to use preAApp by solving ``Rosenbrock`` problem

    This Source Code Form is subject to the terms of the GNU Affero General
    Public License v3.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at https://www.gnu.org/licenses/agpl-3.0.en.html.

    Author(s): Ye Ji <jiyess@outlook.com>
*/

#include <iostream>
//! [Include namespace]
#include "preAA.h"

using namespace preAApp;
//! [Include namespace]

Residual_t residual = [](VectorX const &u) {
  auto n = u.size();

  VectorX F(n);

  for(int i = 0; i < n; i += 2) {
    double t1 = 1.0 - u(i);
    double t2 = 10 * (u(i + 1) - u(i) * u(i));
    F(i + 1) = 20 * t2;
    F(i) = -2.0 * (u(i) * F(i + 1) + t1);
  }
  return F;
};

Jacobian_t jacobian = [](VectorX const &u) {
  auto n = u.size();

  ColMajorSparseMatrix hessian(n, n);
  hessian.reserve(2 * n - 2);

  for (int i = 0; i < n; i += 2) {
    hessian.coeffRef(i, i) = 1200.0*u(i)*u(i) - 400.0*u(i + 1) + 2.0;
    hessian.coeffRef(i + 1, i) = -400.0 * u(i);
    hessian.coeffRef(i, i + 1) = -400.0 * u(i);
    hessian.coeffRef(i + 1, i + 1) = 200.0; 
  }

  hessian.finalize();
  return hessian;
};

Jacobian_t jacobianNumerical = [](VectorX const &u) {
  auto n = u.size();

  ColMajorSparseMatrix hessian(n, n);

  Scalar eps = 1e-8;
  VectorX currentResidual = residual(u);
  VectorX forwardResidual, diffVec;
  VectorX uu = u;

  for (auto i=0; i<n; ++i) {
    uu(i) += eps;
    forwardResidual = residual(uu);
    diffVec = (forwardResidual - currentResidual) / eps;
    for (auto j=0; j<n; ++j) {
      hessian.coeffRef(i, j) = diffVec(j);
    }
    uu(i) -= eps;
  }

  return hessian;
};

int main(int argc, char *argv[]) {
  int m = 3, n = 200;

  preAAParam<> param;
  param.m = m;
  param.usePreconditioning = true;
  param.updatePreconditionerStep = 1;
  param.epsilon = 1e-10;

  AndersonAcceleration<> AASolver(param);

  VectorX initialGuess = VectorX::Zero(n);

  measureTime([&]() {
    VectorX sol = AASolver.compute(initialGuess, residual, jacobian);
  });

}