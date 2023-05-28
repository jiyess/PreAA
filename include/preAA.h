/** @file preAA.h

    @brief Anderson acceleration solver and its (preconditioned) variants

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): Ye Ji (jiyess@outlook.com)
*/

#ifndef PREAAPP_INCLUDE_PREAA_H_
#define PREAAPP_INCLUDE_PREAA_H_

#pragma once

//#include <Eigen/Core>
//#include <Eigen/Sparse>
#include "Types.h"

/**  
 *  Runtime assertions which display a message  
 *
 */
#ifndef NDEBUG
#   define DEBUG_ASSERT(cond, message) do if(!(cond)) {std::cerr          \
       <<"Assert `"<<#cond<<"` "<<message<<"\n"<<__FILE__<<", line "\
       <<__LINE__<<" ("<<__FUNCTION__<<")"<<std::endl;                    \
       throw std::logic_error("DEBUG_ASSERT"); } while(false)
#else
#   define DEBUG_ASSERT(condition, message)
#endif

const Scalar EPSILON = 1e-14;
typedef std::function<VectorX(VectorX const &)> Residual_t;
typedef std::function<ColMajorSparseMatrix(VectorX const &)> Jacobian_t;

namespace preAApp {

/// @brief Anderson acceleration solver and its (preconditioned) variants
///
/// \ingroup Solver
template<typename T>
class AndersonAcceleration {

 public:
  AndersonAcceleration() = default;

  explicit AndersonAcceleration(int m)
      : m_m(m) {
    DEBUG_ASSERT(m > 0, "m should be greater than 0");
  }

  AndersonAcceleration(int m,
                       int maxIter,
                       T tolerance)
      : m_m(m),
        m_maxIter(maxIter),
        m_tolerance(tolerance) {
    DEBUG_ASSERT(m >= 0, "m should be greater than 0");
  }

  AndersonAcceleration(int m,
                       int maxIter,
                       T tolerance,
                       int updatePreconditionerStep)
      : m_m(m),
        m_maxIter(maxIter),
        m_tolerance(tolerance),
        m_updatePreconditionerStep(updatePreconditionerStep) {
    DEBUG_ASSERT(m >= 0, "m should be greater than 0");
  }

  void setUpdatePreconditionerStep(const int &updatePreconditionerStep) {
    m_updatePreconditionerStep = updatePreconditionerStep;
  };

  void usePreconditioningON() { m_usePreconditioning = true; }
  void usePreconditioningOFF() { m_usePreconditioning = false; }

  // perform Anderson acceleration iteration
  const VectorX &compute(const VectorX &u0, const Residual_t &F,
                         const Jacobian_t &Jacobian) {
    m_solution = u0;
    init();

    printSolverInfo();

    // Iteration 0
    updateG(F, Jacobian);

    if (m_currResidual < m_tolerance) {
      printf("You are lucky, the initial guess is exactly the solution.\n\n\n");
      return m_solution;
    } else {
      m_solution = m_currentG;
    }

    if (m_printInfo) {
      printf("Iter.      ||F(x)|| \n");
      printIterationInfo();
    }

    // Start iteration
    m_iter = 1;
    if (m_m == 0) {
      // Picard iteration
      PicardIteration(F, Jacobian);
    } else {
      // Anderson acceleration iteration
      AAIteration(F, Jacobian);
    }
    return m_solution;
  }

  void printIterInfoON() { m_printInfo = true; }
  void printIterInfoOFF() { m_printInfo = false; }

 private:
  // m_m: number of previous iterations used
  // m_d: dimension of variables
  // u0: initial guess
  void init() {
    m_dim = m_solution.size();
    m_iter = 0;
    m_columnIndex = 0;

    m_solution.resize(m_dim);
    m_currentF.resize(m_dim);
    m_prevdG.resize(m_dim, m_m);
    m_prevdF.resize(m_dim, m_m);

    // TODO: if use preconditioner
    if (m_usePreconditioning) {m_preconditioner.resize(m_dim, m_dim);}
    if (m_updatePreconditionerStep<=0) {m_updatePreconditionerStep=10;}

    m_normalEquationMatrix.resize(m_m, m_m);
    m_alpha.resize(m_m);
    m_scaledF.resize(m_m);
  }

  void printSolverInfo() {
    printf("\n SOLVER: parameter settings are as follows... \n");
    printf("depth                       =     %d\n", m_m);
    printf("use preconditioner          =     %d\n", m_usePreconditioning);
    printf("update preconditioner step  =     %d\n\n",
           m_updatePreconditionerStep);
  }

  /// update fixed point function
  void updateG(const Residual_t &F,
               const Jacobian_t &Jacobian) {
    if (m_usePreconditioning) {
      // preconditioning version
      if (!(m_iter % m_updatePreconditionerStep)) {
        m_preconditioner = Jacobian(m_solution);
        m_linearSolverPreconditioning.compute(m_preconditioner);
      }
      VectorX residual = F(m_solution);
      m_currentF = -m_linearSolverPreconditioning.solve(residual);
      m_currResidual = residual.norm();
    } else {
      // no preconditioning version
      m_currentF = F(m_solution);
      m_currResidual = m_currentF.norm();
    }

    m_currentG = m_currentF + m_solution;
  }

  inline void printIterationInfo() {
    printf(" %d         %.4e\n", m_iter, m_currResidual);
  }

  void PicardIteration(const Residual_t &F,
                       const Jacobian_t &Jacobian) {
    if (m_usePreconditioning) {
      // preconditioning version
      while (m_iter < m_maxIter && m_currResidual > m_tolerance) {
        if (!(m_iter % m_updatePreconditionerStep)) {
          m_preconditioner = Jacobian(m_solution);
          m_linearSolverPreconditioning.compute(m_preconditioner);
        }
        VectorX residual = F(m_solution);
        m_currentF = -m_linearSolverPreconditioning.solve(residual);
        m_currResidual = residual.norm();

        m_solution += m_currentF;
        if (m_printInfo) { printIterationInfo(); }
        m_iter++;
      }
    } else {
      while (m_iter < m_maxIter && m_currResidual > m_tolerance) {
        m_currentF = F(m_solution);
        m_currResidual = m_currentF.norm();

        m_solution += m_currentF;
        if (m_printInfo) { printIterationInfo(); }
        m_iter++;
      }
    }
  }

  void AAIteration(const Residual_t &F, const Jacobian_t &Jacobian) {

    m_prevdF.col(0) = -m_currentF;
    m_prevdG.col(0) = -m_currentG;

    while (m_iter < m_maxIter && m_currResidual > m_tolerance) {

      updateG(F, Jacobian);

      if (m_printInfo) { printIterationInfo(); }

      m_prevdF.col(m_columnIndex) += m_currentF;
      m_prevdG.col(m_columnIndex) += m_currentG;

      T scale = std::max(EPSILON, m_prevdF.col(m_columnIndex).norm());
      m_scaledF(m_columnIndex) = scale;
      m_prevdF.col(m_columnIndex) /= scale;

      int m_k = std::min(m_m, m_iter);

      if (m_k == 1) {
        m_alpha(0) = 0;
        T dF_squaredNorm = m_prevdF.col(m_columnIndex).squaredNorm();
        m_normalEquationMatrix(0, 0) = dF_squaredNorm;
        T dF_norm = std::sqrt(dF_squaredNorm);

        // For better numerical stability
        if (dF_norm > EPSILON) {
          // compute alpha = (dF * F) / (dF * dF)
          m_alpha(0) = (m_prevdF.col(m_columnIndex) / dF_norm).dot(
              m_currentF / dF_norm);
        }
      } else {
        // Update the normal equation matrix
        // for the column and row corresponding to the new dF column.
        // note: only one column and one row are needed to be updated.
        VectorX new_inner_prod = (m_prevdF.col(m_columnIndex).transpose()
            * m_prevdF.block(0, 0, m_dim, m_k)).transpose();
        m_normalEquationMatrix.block(m_columnIndex, 0, 1, m_k) =
            new_inner_prod.transpose();
        m_normalEquationMatrix.block(0, m_columnIndex, m_k, 1) =
            new_inner_prod;

        // Solve normal equation: A^{T} A x = A^{T} b
        m_linearSolver.compute(m_normalEquationMatrix.block(0, 0, m_k, m_k));
        m_alpha.head(m_k) = m_linearSolver.solve(
            m_prevdF.block(0, 0, m_dim, m_k).transpose() * m_currentF);
      }
      // Update the current solution (x) using the rescaled alpha
      m_solution = m_currentG - m_prevdG.block(0, 0, m_dim, m_k) *
          ((m_alpha.head(m_k).array()
              / m_scaledF.head(m_k).array()).matrix());

      // TODO: add mixing parameter \beta
//        m_solution = (1-m_beta) * m_solution + m_beta * m_currentG - (
//            m_prevdG.block(0, 0, m_dim, m_k) * ( m_beta *
//                (m_alpha.head(m_k).array()/m_scaledF.head(m_k).array())
//                .matrix()) -
//                (1-m_beta) *
//            );

      m_columnIndex = (m_columnIndex + 1) % m_m;
      m_prevdF.col(m_columnIndex) = -m_currentF;
      m_prevdG.col(m_columnIndex) = -m_currentG;

      m_iter++;

      // TODO: if record history list
      m_residualList.push_back(m_currResidual);
    }
  }

//    Eigen::CompleteOrthogonalDecomposition<MatrixXX> m_linearSolver;
  Eigen::SparseLU<ColMajorSparseMatrix> m_linearSolverPreconditioning;
// 候选求解器: Eigen::PartialPivLU, Eigen::FullPivLU, Eigen::HouseholderQR,
//      Eigen::ColPivHouseholderQR, Eigen::JacobiSVD
  Eigen::FullPivLU<MatrixXX> m_linearSolver;
  ColMajorSparseMatrix m_preconditioner;
  VectorX m_solution;
  VectorX m_currentF;
  VectorX m_currentG;
  MatrixXX m_prevdG;
  MatrixXX m_prevdF;

  // Normal equations matrix for the computing alpha
  MatrixXX m_normalEquationMatrix;
  // alpha value computed from normal equations
  VectorX m_alpha;
  // The scaling factor for each column of prev_dF
  VectorX m_scaledF;

  T m_currResidual = std::numeric_limits<T>::max();
  T m_beta = 1.0;
  int m_dim;  // Dimension of variables
  int m_iter;    // Iteration count since initialization
  int m_columnIndex; // Index for history matrix column to store the next value

  bool m_usePreconditioning = false;
  bool m_printInfo = true;

  std::vector<T> m_residualList;

  // TODO: use gsOptionsList?
  // Number of previous iterates used for Andreson Acceleration
  int m_m = 5; // depth (windows size), typically small m <= 10
  int m_maxIter = 1e3; // maximum iteration
  T m_tolerance = 1e-5; // tolerance for convergence test

  // update the preconditioner every updatePreconditionerStep step
  int m_updatePreconditionerStep = -1;
};
}

#endif //PREAAPP_INCLUDE_PREAA_H_
