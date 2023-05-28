/** @file preAAParam.h

    @brief Parameters to control the preconditioned AA  algorithm

    This Source Code Form is subject to the terms of the GNU Affero General
    Public License v3.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at https://www.gnu.org/licenses/agpl-3.0.en.html.

    Author(s): Ye Ji <jiyess@outlook.com>
*/

#ifndef PREAAPP_INCLUDE_PREAAPARAM_H_
#define PREAAPP_INCLUDE_PREAAPARAM_H_

#include <stdexcept>  // std::invalid_argument

///
/// Parameters to control the preconditioned Anderson Acceleration algorithm
///
template <typename Scalar = double>
class preAAParam {
 public:
  ///
  /// Depth: The number of previous iterates used for Andreson Acceleration.
  /// Typically in application \ref m is small, say <= 10.
  /// There is usually little advantage in large \ref m.
  /// The default value is \c 5.
  /// Large values result in excessive computing time.
  ///
  int m;
  ///
  /// Absolute tolerance for convergence test.
  /// This parameter determines the absolute accuracy \f$\epsilon_{abs}\f$
  /// with which the solution is to be found. A minimization terminates when
  /// \f$||F|| < \epsilon_{abs}\f$, where \f$||\cdot||\f$ denotes the
  /// Euclidean (L2) norm. The default value is \c 1e-5.
  ///
  Scalar epsilon;
  ///
  /// Relative tolerance for convergence test.
  /// This parameter determines the relative accuracy \f$\epsilon_{rel}\f$
  /// with which the solution is to be found. A minimization terminates when
  /// \f$||F|| < \max\{\epsilon_{abs}, \epsilon_{rel}||x||\}\f$,
  /// where \f$||\cdot||\f$ denotes the Euclidean (L2) norm.
  /// The default value is \c 1e-5. 
  /// 
  Scalar epsilon_rel;
  ///
  /// The maximum number of iterations.
  /// The iteration process is terminated when the iteration count exceeds
  /// this parameter. Setting this parameter to zero continues an iteration
  /// process until a convergence or error. The default value is \c 0.
  ///
  int max_iterations;
  ///
  /// The step to update the preconditioner.
  /// update the preconditioner every updatePreconditionerStep step
  ///
  int updatePreconditionerStep;
  ///
  /// Mixing or damping parameter.
  /// It plays exactly the same role as it does for Picard iteration.
  /// One could vary \ref beta as the iteration.
  ///
  Scalar beta;
  ///
  /// Use preconditioning or not?
  ///
  bool usePreconditioning;

 public:
  ///
  /// Constructor for L-BFGS parameters.
  /// Default values for parameters will be set when the object is created.
  ///
  preAAParam() {
    // clang-format off
    m = 5;
    epsilon = Scalar(1e-5);
    epsilon_rel = Scalar(1e-5);
    max_iterations = 100;
    updatePreconditionerStep = 10;
    beta = Scalar(1.0);
    usePreconditioning = false;
    // clang-format on
  }

  ///
  /// Checking the validity of preconditioned AA parameters.
  /// An `std::invalid_argument` exception will be thrown if some parameter
  /// is invalid.
  ///
  inline void check_param() const {
    if (m < 0)
      throw std::invalid_argument("'m' must be non-negative");
    if (epsilon < 0)
      throw std::invalid_argument("'epsilon' must be non-negative");
    if (epsilon_rel < 0)
      throw std::invalid_argument("'epsilon_rel' must be non-negative");
    if (max_iterations < 0)
      throw std::invalid_argument("'max_iterations' must be non-negative");
  }
};

#endif //PREAAPP_INCLUDE_PREAAPARAM_H_
