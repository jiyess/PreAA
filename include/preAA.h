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
using Residual_t = std::function<VectorX(const VectorX &)>;
using Jacobian_t = std::function<ColMajorSparseMatrix(const VectorX &)>;

namespace preAApp {

template<typename Func>
inline void measureTime(Func func) {
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
    checkParam();
  }

  /// perform Anderson acceleration iteration
  const VectorX &compute(const VectorX &u0, const Residual_t &F,
                         const Jacobian_t &Jacobian) {
    // print the current solver information
    printSolverInfo();

    // initialize the solver and preallocate memory
    initialize(u0);

    // Iteration 0
    updateG(F, Jacobian);

    // check if the initial guess is the solution
    if (m_currResidualNorm < m_param.epsilon) {
      printf("You are lucky, the initial guess is exactly the solution.\n\n\n");
      return u0;
    } else {
      m_solution = m_currentG;
    }

    // print iteration information
    if (m_printInfo) {
      printf("Iter.      ||F(x)|| \n");
      printIterationInfo();
    }

    // Start iteration
    startIteration(F, Jacobian);

    return m_solution;
  }

  inline void enableIterationInfoPrinting() { m_printInfo = true; }
  inline void disableIterationInfoPrinting() { m_printInfo = false; }

 private:
  inline void initialize(const VectorX &u0) {
    m_solution = u0;
    m_dim = m_solution.size();
    m_iter = 0;
    m_columnIndex = 0;

    m_currentF.resize(m_dim);
    m_prevdG.resize(m_dim, m_param.m);
    m_prevdF.resize(m_dim, m_param.m);

    m_normalEquationMatrix.resize(m_param.m, m_param.m);
    m_alpha.resize(m_param.m);
    m_scaledF.resize(m_param.m);

    if (m_param.usePreconditioning) { m_preconditioner.resize(m_dim, m_dim); }
  }

  inline void checkParam() {
    m_param.check_param();
  }

  inline void printSolverInfo() {
    printf("\nAnderson Acceleration SOLVER: parameter settings... \n");
    printf("depth                       =     %d\n", m_param.m);
    printf("use preconditioner          =     %d\n",
           m_param.usePreconditioning);
    printf("update preconditioner step  =     %d\n\n",
           m_param.updatePreconditionerStep);
  }

  inline void startIteration(const Residual_t &F, const Jacobian_t &Jacobian) {
    m_iter = 1;
    if (m_param.m == 0) {
      performPicardIteration(F, Jacobian);
    } else {
      performAAIteration(F, Jacobian);
    }
  }

  inline void performPicardIteration(const Residual_t &F,
                                     const Jacobian_t &Jacobian) {
    while (m_iter < m_param.max_iterations
        && m_currResidualNorm > m_param.epsilon) {
      updateG(F, Jacobian);

      // update the solution
      m_solution = m_currentG;

      printIterationInfo();

      ++m_iter;
      trackIterationInfo();
    }
  }

  inline void performAAIteration(const Residual_t &F,
                                 const Jacobian_t &Jacobian) {
    m_prevdF.col(0) = -m_currentF;
    m_prevdG.col(0) = -m_currentG;

    while (m_iter < m_param.max_iterations &&
        m_currResidualNorm > m_param.epsilon) {
      updateG(F, Jacobian);

      printIterationInfo();

      updatePrevdFAndPrevdG();

      // update alpha and solution (compute normal equation)
      updateAlpha();
      updateSolution();

      // update the column indices
      updateColumnIndices();

      ++m_iter;
      trackIterationInfo();
    }
  }

  /// update fixed point function
  inline void updateG(const Residual_t &F, const Jacobian_t &Jacobian) {
    if (m_param.usePreconditioning) {
      updateGWithPreconditioning(F, Jacobian);
    } else {
      updateGWithoutPreconditioning(F);
    }
    m_currentG = m_currentF + m_solution;
  }

  /// update fixed point function with preconditioning
  inline void updateGWithPreconditioning(const Residual_t &F,
                                         const Jacobian_t &Jacobian) {
    if (!(m_iter % m_param.updatePreconditionerStep)) {
      m_preconditioner = Jacobian(m_solution);
      m_linearSolverPreconditioning.compute(m_preconditioner);
    }
    VectorX residual = F(m_solution);
    m_currResidualNorm = residual.norm();

    m_currentF = -m_linearSolverPreconditioning.solve(residual);
  }

  /// update fixed point function without preconditioning
  inline void updateGWithoutPreconditioning(const Residual_t &F) {
    m_currentF = F(m_solution);
    m_currResidualNorm = m_currentF.norm();
  }

  inline void printIterationInfo() {
    if (m_printInfo) {
      printf(" %d         %.4e\n", m_iter, m_currResidualNorm);
    }
  }

  inline void updatePrevdFAndPrevdG() {
    m_prevdF.col(m_columnIndex) += m_currentF;
    m_prevdG.col(m_columnIndex) += m_currentG;

    // scale previous dF for better numerical stability
    Scalar scale = std::max(EPSILON, m_prevdF.col(m_columnIndex).norm());
    m_scaledF(m_columnIndex) = scale;
    m_prevdF.col(m_columnIndex) /= scale;
  }

  /// update the coefficients \f$ alpha_i \f$ by solving a Least-Square problem
  inline void updateAlpha() {
    // compute m_mk
    m_mk = std::min(m_param.m, m_iter);

    if (m_mk == 1) {
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
          * m_prevdF.block(0, 0, m_dim, m_mk)).transpose();
      m_normalEquationMatrix.block(m_columnIndex, 0, 1, m_mk) =
          new_inner_prod.transpose();
      m_normalEquationMatrix.block(0, m_columnIndex, m_mk, 1) =
          new_inner_prod;

      // Solve normal equation: A^{T} A x = A^{T} b
      m_linearSolver.compute(m_normalEquationMatrix.block(0, 0, m_mk, m_mk));
      m_alpha.head(m_mk) = m_linearSolver.solve(
          m_prevdF.block(0, 0, m_dim, m_mk).transpose() * m_currentF);
    }
  }

  /// update the solution
  inline void updateSolution() {
    // Update the current solution (x) using the rescaled alpha

    m_solution = m_currentG - m_prevdG.block(0, 0, m_dim, m_mk) *
        ((m_alpha.head(m_mk).array()
            / m_scaledF.head(m_mk).array()).matrix());

    // TODO: add mixing parameter \beta
//    if (m_useMixingBeta){
//      m_solution = ((1-m_param.beta) * (m_solution -
//          m_prevSolution.block(0, 0, m_dim, m_mk)) + m_param.beta *
//              (m_currentG - m_prevdG.block(0, 0, m_dim, m_mk)) *  ((m_alpha
//              .head(m_mk).array() / m_scaledF.head(m_mk).array()).matrix()));
//    }
//    else {
//      m_solution = m_currentG - m_prevdG.block(0, 0, m_dim, m_mk) *
//          ((m_alpha.head(m_mk).array()
//              / m_scaledF.head(m_mk).array()).matrix());
//    }
  }

  void updateColumnIndices() {
    m_columnIndex = (m_columnIndex + 1) % m_param.m;
    m_prevdF.col(m_columnIndex) = -m_currentF;
    m_prevdG.col(m_columnIndex) = -m_currentG;
  }

  inline void trackIterationInfo() {
    if (m_trackResidualNorm) { m_residualList.push_back(m_currResidualNorm); }
  }

  // Linear solver for the Least-Square problem
  Eigen::FullPivLU<MatrixXX> m_linearSolver;
  // Parameters
  const preAAParam<Scalar> m_param;
  int m_dim = -1;
  // Iteration
  int m_iter = 0;
  int m_mk = 0;
  int m_columnIndex = 0;
  // Residual
  Scalar m_currResidualNorm;
  // Solution
  VectorX m_solution;
  VectorX m_currentF;
  VectorX m_currentG;
  // Anderson extrapolation
  MatrixXX m_prevdG;
  MatrixXX m_prevdF;
  MatrixXX m_normalEquationMatrix;
  VectorX m_alpha;
  VectorX m_scaledF;
  // Preconditioning
  ColMajorSparseMatrix m_preconditioner;
  Eigen::SparseLU<ColMajorSparseMatrix> m_linearSolverPreconditioning;
  // Print information
  bool m_printInfo = true;

  // track residual norm history
  bool m_trackResidualNorm = false;
  std::vector<Scalar> m_residualList;
};

}

#endif //PREAAPP_INCLUDE_PREAA_H_
