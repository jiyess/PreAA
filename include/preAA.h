/** @file preAA.h

    @brief Anderson acceleration solver and its (preconditioned) variants

    This Source Code Form is subject to the terms of the GNU Affero General
    Public License v3.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at https://www.gnu.org/licenses/agpl-3.0.en.html.

    Author(s): Ye Ji <jiyess@outlook.com>
*/

#ifndef PREAAPP_INCLUDE_PREAA_H_
#define PREAAPP_INCLUDE_PREAA_H_

#pragma once

#include "preAAParam.h"
#include "Types.h"

const Scalar EPSILON = 1e-14;
typedef std::function<VectorX(VectorX const &)> Residual_t;
typedef std::function<ColMajorSparseMatrix(VectorX const &)> Jacobian_t;

namespace preAApp {

template<typename Func>
void measureTime(Func func) {
  auto beforeTime = std::chrono::high_resolution_clock::now();

  func();  // Call the function

  auto afterTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double>(afterTime - beforeTime).count();
  printf("\nTime passed by %.8f sec. \n", duration);
}

/// @brief Anderson acceleration solver and its (preconditioned) variants
///
/// \ingroup Solver
template<typename Scalar = double>
class AndersonAcceleration {
 public:

  explicit AndersonAcceleration(const preAAParam<Scalar> &param) :
      m_param(param) {
    m_param.check_param();
  }

  // perform Anderson acceleration iteration
  const VectorX &compute(const VectorX &u0, const Residual_t &F,
                         const Jacobian_t &Jacobian) {
    m_solution = u0;
    init();

    printSolverInfo();

    // Iteration 0
    updateG(F, Jacobian);

    if (m_currResidualNorm < m_param.epsilon) {
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
    // TODO: consider the code structure carefully!
    if (m_param.m == 0) {
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
  inline void init() {
    m_dim = m_solution.size();
    m_iter = 0;
    m_columnIndex = 0;

    m_solution.resize(m_dim);
    m_currentF.resize(m_dim);
    m_prevdG.resize(m_dim, m_param.m);
    m_prevdF.resize(m_dim, m_param.m);

    m_normalEquationMatrix.resize(m_param.m, m_param.m);
    m_alpha.resize(m_param.m);
    m_scaledF.resize(m_param.m);

    if (m_param.usePreconditioning) { m_preconditioner.resize(m_dim, m_dim); }
  }

  void printSolverInfo() {
    printf("\nAnderson Acceleration SOLVER: parameter settings... \n");
    printf("depth                       =     %d\n", m_param.m);
    printf("use preconditioner          =     %d\n",
           m_param.usePreconditioning);
    printf("update preconditioner step  =     %d\n\n",
           m_param.updatePreconditionerStep);
  }

  /// update fixed point function
  void updateG(const Residual_t &F,
               const Jacobian_t &Jacobian) {
    if (m_param.usePreconditioning) {
      // preconditioning version
      if (!(m_iter % m_param.updatePreconditionerStep)) {
        m_preconditioner = Jacobian(m_solution);
        m_linearSolverPreconditioning.compute(m_preconditioner);
      }
      VectorX residual = F(m_solution);
      m_currentF = -m_linearSolverPreconditioning.solve(residual);
      m_currResidualNorm = residual.norm();
    } else {
      // no preconditioning version
      m_currentF = F(m_solution);
      m_currResidualNorm = m_currentF.norm();
    }

    m_currentG = m_currentF + m_solution;
  }

  inline void printIterationInfo() {
    printf(" %d         %.4e\n", m_iter, m_currResidualNorm);
  }

  void PicardIteration(const Residual_t &F,
                       const Jacobian_t &Jacobian) {
    if (m_param.usePreconditioning) {
      // preconditioning version
      while (m_iter < m_param.max_iterations
          && m_currResidualNorm > m_param.epsilon) {
        if (!(m_iter % m_param.updatePreconditionerStep)) {
          m_preconditioner = Jacobian(m_solution);
          m_linearSolverPreconditioning.compute(m_preconditioner);
        }
        VectorX residual = F(m_solution);
        m_currentF = -m_linearSolverPreconditioning.solve(residual);
        m_currResidualNorm = residual.norm();

        m_solution += m_currentF;
        if (m_printInfo) { printIterationInfo(); }
        m_iter++;
      }
    } else {
      while (m_iter < m_param.max_iterations
          && m_currResidualNorm > m_param.epsilon) {
        m_currentF = F(m_solution);
        m_currResidualNorm = m_currentF.norm();

        m_solution += m_currentF;
        if (m_printInfo) { printIterationInfo(); }
        m_iter++;
      }
    }
  }

  void AAIteration(const Residual_t &F, const Jacobian_t &Jacobian) {

    m_prevdF.col(0) = -m_currentF;
    m_prevdG.col(0) = -m_currentG;

    while (m_iter < m_param.max_iterations
        && m_currResidualNorm > m_param.epsilon) {

      updateG(F, Jacobian);

      if (m_printInfo) { printIterationInfo(); }

      m_prevdF.col(m_columnIndex) += m_currentF;
      m_prevdG.col(m_columnIndex) += m_currentG;

      Scalar scale = std::max(EPSILON, m_prevdF.col(m_columnIndex).norm());
      m_scaledF(m_columnIndex) = scale;
      m_prevdF.col(m_columnIndex) /= scale;

      int m_k = std::min(m_param.m, m_iter);

      if (m_k == 1) {
        m_alpha(0) = 0;
        Scalar dF_squaredNorm = m_prevdF.col(m_columnIndex).squaredNorm();
        m_normalEquationMatrix(0, 0) = dF_squaredNorm;
        Scalar dF_norm = std::sqrt(dF_squaredNorm);

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

      m_columnIndex = (m_columnIndex + 1) % m_param.m;
      m_prevdF.col(m_columnIndex) = -m_currentF;
      m_prevdG.col(m_columnIndex) = -m_currentG;

      m_iter++;

      if (m_trackResidualNorm) { m_residualList.push_back(m_currResidualNorm); }
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

  const preAAParam<Scalar> &m_param;

  // Normal equations matrix for the computing alpha
  MatrixXX m_normalEquationMatrix;
  // alpha value computed from normal equations
  VectorX m_alpha;
  // The scaling factor for each column of prev_dF
  VectorX m_scaledF;
  // The norm of current residual
  Scalar m_currResidualNorm = std::numeric_limits<Scalar>::max();

  int m_dim = -1;  // Dimension of variables
  int m_iter = 0;    // Iteration count since initialization
  int m_columnIndex = 0; // Index for history matrix column to store the next
  // value

  bool m_printInfo = true;
  bool m_trackResidualNorm = false;
  std::vector<Scalar> m_residualList;
};
}

#endif //PREAAPP_INCLUDE_PREAA_H_
