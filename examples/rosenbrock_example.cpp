/** @file AndersonAcceleration_example.cpp

    @brief Tutorial on how to use Anderson acceleration and its (preconditioned) variants

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): Ye Ji (jiyess@outlook.com)
*/

//! [Include namespace]
#include "preAA.h"

using namespace preAApp;
//! [Include namespace]

Residual_t residual = [](VectorX const &u) {
  auto n = u.size();

  VectorX F(n);
//    G[0] = -0.5* pow(u[0]-u[1], 3) + 1.0;
//    G[1] = -0.5* pow(u[1]-u[0], 3);

  F(0) = cos(0.5 * (u(0)+u(1)));
  F(1) = F(0) + 1e-8 * sin(u(1)*u(1));

//    F[0] = pow(u(0),2);
//    F[1] = pow(u(1),2);

//    real_t b = 1;
//    F(0) = 2*(u(0)-1) - 4 * b * (u(1) - pow(u(0),2))*u(0);
//    F(1) = 2 * b * (u(1)-pow(u(0), 2));

//    for (auto i=0; i<n; i+=2){
//      double t1 = 1.0 - u[i];
//      double t2 = 10 * (u[i + 1] - u[i] * u[i]);
//      F[i + 1] = 20 * t2;
//      F[i]     = -2.0 * (u[i] * F[i + 1] + t1);
//    }

//    F(0) = sin(u(0)) + u(1) - 1.0;
//    F(1) = u(0) + cos(u(1)) - 2.0;
  return F;
};

Jacobian_t jacobian = [](VectorX const &u) {
  auto n = u.size();

  ColMajorSparseMatrix jac(n,n);
//    VectorX F(n);

//    jac(0,0) = u(0);
//    jac(1,0) = 0;
//    jac(0,1) = 0;
//    jac(1,1) = u(1);

//    jac.insert(0,0) = -0.5*sin(0.5*(u(0)+u(1)));
//    jac.insert(1,0) = -0.5*sin(0.5*(u(0)+u(1)));
//    jac.insert(0,1) = -0.5*sin(0.5*(u(0)+u(1)));
//    jac.insert(1,1) = -0.5*sin(0.5*(u(0)+u(1))) + 2e-8*u(1)*cos(pow(u(1),2));
//    jac.makeCompressed();

  double val = -0.5*sin(0.5*(u(0)+u(1)));
  jac.coeffRef(0,0) = val;
  jac.coeffRef(1,0) = val;
  jac.coeffRef(0,1) = val;
  jac.coeffRef(1,1) = val+2e-8*u(1)*cos(pow(u(1),2));
  jac.makeCompressed();

//      jac.setIdentity();
//    jac.setIdentity();

  return jac;
};

int main(int argc, char *argv[])
{
  int m = 0;

  AndersonAcceleration<double> AASolver(m, 1000, 1e-10);
  AASolver.usePreconditioningON();
//  AASolver.usePreconditioningOFF();
//  AASolver.setUpdatePreconditionerStep(2);
//  AASolver.printIterInfoOFF();

  VectorX initialGuess = VectorX::Ones(2);

  measureTime([&]() {
    VectorX sol = AASolver.compute(initialGuess, residual, jacobian);
  });

}