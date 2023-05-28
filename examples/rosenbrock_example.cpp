//
// Created by 纪野 on 2023/5/22.
//

/** @file AndersonAcceleration_example.cpp

    @brief Tutorial on how to use Anderson acceleration and its (preconditioned) variants

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): Ye Ji
*/

//! [Include namespace]
#include <iostream>
#include <chrono>
#include "preAA.h"

using namespace preAApp;
//! [Include namespace]

typedef std::function<VectorX(VectorX const &)> Residual_t;
typedef std::function<ColMajorSparseMatrix(VectorX const &)> Jacobian_t;

Residual_t residual = [](VectorX const &u) {

  int n = u.size();

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
  int n = u.size();

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
//  G[1]=-0.5*(u[1]-u[2])^3+1.0
//  G[2]=-0.5*(u[2]-u[1])^3

  int m = 0;
  AndersonAcceleration<double> AASolver(m, 1000, 1e-10);
  AASolver.usePreconditioningON();
//  AASolver.usePreconditioningOFF();
//  AASolver.setUpdatePreconditionerStep(2);
//  AASolver.printIterInfoOFF();

  int n = 2;
  VectorX initialGuess(n);
  initialGuess.setOnes();

  auto begin = std::chrono::high_resolution_clock::now();
  VectorX sol = AASolver.compute(initialGuess, residual, jacobian);

  // Stop measuring time and calculate the elapsed time
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

  std::cout << "\n Time passed by " << elapsed.count() * 1e-9 << " sec. \n";
}

//#include <iostream>
//#include <cmath>
//
//// 定义非线性方程组的函数
//double f1(double x, double y) {
//  return sin(x) + y - 1.0;
//}
//
//double f2(double x, double y) {
//  return x + cos(y) - 2.0;
//}
//
//// Picard迭代求解非线性方程组
//void picardIteration(double& x, double& y, int maxIterations, double tolerance) {
//  double x0 = x;
//  double y0 = y;
//
//  for (int i = 0; i < maxIterations; ++i) {
//    double newX = f1(x0, y0);
//    double newY = f2(x0, y0);
//
//    if (std::abs(newX - x0) < tolerance && std::abs(newY - y0) < tolerance) {
//      x = newX;
//      y = newY;
//      return;
//    }
//
//    std::cout << "iter = " << i << ", res = " << newX*newX + newY*newY << "\n";
//
//    x0 = newX;
//    y0 = newY;
//  }
//
//  // 如果达到最大迭代次数仍未收敛，则输出错误信息
//  std::cout << "Picard迭代未收敛" << std::endl;
//}
//
//int main() {
//  // 设置初始猜测值
//  double x = 0.0;
//  double y = 0.0;
//
//  // 设置最大迭代次数和容差
//  int maxIterations = 100;
//  double tolerance = 1e-6;
//
//  // 使用Picard迭代求解非线性方程组
//  picardIteration(x, y, maxIterations, tolerance);
//
//  // 输出结果
//  std::cout << "解的近似值为：x = " << x << ", y = " << y << std::endl;
//
//  return 0;
//}
