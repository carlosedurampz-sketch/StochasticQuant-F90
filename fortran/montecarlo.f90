!===============================================================================
! StochasticQuant-F90 — montecarlo.f90
! Fortran 90 numerical core for portfolio tail-risk Monte Carlo.
!
! What it does
!   * Box-Muller generation of standard normals from Fortran's RNG
!     (seeded, reproducible: same seed -> same draws).
!   * Cholesky decomposition of a correlation / covariance matrix.
!   * Correlated GBM terminal values (Euler-Maruyama on log-price).
!   * Vasicek single-factor correlated-default simulation.
!   * Merton structural simulation for N obligors with a full
!     asset-correlation matrix (Cholesky).
!   * Antithetic variates (Z and -Z averaged) and a control variate
!     (European call under Black-Scholes) to cut Monte Carlo variance.
!
! Build (see ../Makefile):
!   f2py -c -m montecarlo montecarlo.f90
!   -- or --
!   gfortran -O3 -fPIC -shared -o libmontecarlo.so montecarlo.f90
!
! The Python fallback in ../src/fortran_backend.py reproduces the exact
! f2py API (same subroutine names / argument order) with NumPy, so the
! project runs end-to-end on machines without a Fortran compiler.
!===============================================================================
module montecarlo
  implicit none
  private
  public :: randn_array, cholesky, simulate_gbm, simulate_vasicek_losses, &
            simulate_merton_losses, gbm_call_control_variate, &
            black_scholes_call, portfolio_var_cvar

  double precision, parameter :: PI = 3.14159265358979323846d0

contains

  !---------------------------------------------------------------------------
  ! Seed the Fortran RNG deterministically.
  !---------------------------------------------------------------------------
  subroutine seed_rng(seed)
    integer, intent(in) :: seed
    integer :: n, i
    integer, allocatable :: s(:)
    call random_seed(size=n)
    allocate(s(n))
    do i = 1, n
      s(i) = seed + 37 * (i - 1)
    end do
    call random_seed(put=s)
    deallocate(s)
  end subroutine seed_rng

  !---------------------------------------------------------------------------
  ! randn_array(n, seed, z): n i.i.d. N(0,1) draws via Box-Muller.
  !   Box-Muller: given U1,U2 ~ Uniform(0,1),
  !     Z0 = sqrt(-2 ln U1) cos(2 pi U2), Z1 = sqrt(-2 ln U1) sin(2 pi U2).
  !---------------------------------------------------------------------------
  subroutine randn_array(n, seed, z)
    integer, intent(in) :: n, seed
    double precision, intent(out) :: z(n)
    double precision :: u1, u2, r, theta
    integer :: i
    call seed_rng(seed)
    i = 1
    do while (i <= n)
      call random_number(u1)
      call random_number(u2)
      if (u1 <= 0.0d0) u1 = 1.0d-12
      r = sqrt(-2.0d0 * log(u1))
      theta = 2.0d0 * PI * u2
      z(i) = r * cos(theta)
      if (i + 1 <= n) z(i + 1) = r * sin(theta)
      i = i + 2
    end do
  end subroutine randn_array

  !---------------------------------------------------------------------------
  ! cholesky(n, a, l): L such that A = L L^T (lower-triangular).
  !   info = 0 on success, >0 if A is not positive definite.
  !---------------------------------------------------------------------------
  subroutine cholesky(n, a, l, info)
    integer, intent(in) :: n
    double precision, intent(in) :: a(n, n)
    double precision, intent(out) :: l(n, n)
    integer, intent(out) :: info
    integer :: i, j, k
    double precision :: s
    l = 0.0d0
    info = 0
    do i = 1, n
      do j = 1, i
        s = a(i, j)
        do k = 1, j - 1
          s = s - l(i, k) * l(j, k)
        end do
        if (i == j) then
          if (s <= 0.0d0) then
            info = i
            return
          end if
          l(i, j) = sqrt(s)
        else
          l(i, j) = s / l(j, j)
        end if
      end do
    end do
  end subroutine cholesky

  !---------------------------------------------------------------------------
  ! simulate_gbm(n_paths, n_steps, s0, mu, sigma, t, seed, antithetic,
  !              terminal, mean_terminal, std_terminal)
  !   GBM: dS = mu S dt + sigma S dW. Log-price Euler-Maruyama is exact
  !   for the terminal value:
  !     S_T = S0 * exp((mu - sigma^2/2) T + sigma sqrt(T) Z)
  !   With antithetic=.true., each Z is paired with -Z and the payoff
  !   is averaged, which halves the variance for symmetric payoffs.
  !---------------------------------------------------------------------------
  subroutine simulate_gbm(n_paths, n_steps, s0, mu, sigma, t, seed, &
                          antithetic, terminal, mean_terminal, std_terminal)
    integer, intent(in) :: n_paths, n_steps, seed
    logical, intent(in) :: antithetic
    double precision, intent(in) :: s0, mu, sigma, t
    double precision, intent(out) :: terminal(n_paths)
    double precision, intent(out) :: mean_terminal, std_terminal
    double precision, allocatable :: z(:)
    double precision :: dt, drift, diff, s_up, s_dn
    integer :: i, half
    dt = t / dble(n_steps)
    drift = (mu - 0.5d0 * sigma * sigma) * t
    diff = sigma * sqrt(t)
    allocate(z(n_paths))
    call randn_array(n_paths, seed, z)
    if (antithetic) then
      half = n_paths / 2
      do i = 1, half
        s_up = s0 * exp(drift + diff * z(i))
        s_dn = s0 * exp(drift - diff * z(i))
        terminal(i) = 0.5d0 * (s_up + s_dn)
        terminal(half + i) = terminal(i)
      end do
      if (mod(n_paths, 2) == 1) terminal(n_paths) = s0 * exp(drift + diff * z(n_paths))
    else
      do i = 1, n_paths
        terminal(i) = s0 * exp(drift + diff * z(i))
      end do
    end if
    deallocate(z)
    mean_terminal = sum(terminal) / dble(n_paths)
    std_terminal = sqrt(sum((terminal - mean_terminal)**2) / dble(max(n_paths - 1, 1)))
  end subroutine simulate_gbm

  !---------------------------------------------------------------------------
  ! black_scholes_call(s, k, r, sigma, t): analytic European call price.
  !   Used as the control variate: E[payoff] is known, so
  !   payoff_adj = payoff - beta * (control - E[control]).
  !---------------------------------------------------------------------------
  function black_scholes_call(s, k, r, sigma, t) result(price)
    double precision, intent(in) :: s, k, r, sigma, t
    double precision :: price, d1, d2
    d1 = (log(s / k) + (r + 0.5d0 * sigma * sigma) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    price = s * norm_cdf(d1) - k * exp(-r * t) * norm_cdf(d2)
  end function black_scholes_call

  function norm_cdf(x) result(p)
    double precision, intent(in) :: x
    double precision :: p
    p = 0.5d0 * erfc(-x / sqrt(2.0d0))
  end function norm_cdf

  function norm_ppf(p) result(x)
    ! Acklam's rational approximation to Phi^{-1}(p), |error| < 1.15e-9.
    double precision, intent(in) :: p
    double precision :: x, q, r
    double precision, parameter :: a1 = -3.969683028665376e+01, a2 = 2.209460984245205e+02
    double precision, parameter :: a3 = -2.759285984469534e+02, a4 = 1.383577518672690e+02
    double precision, parameter :: a5 = -3.066479806614716e+01, a6 = 2.506628277459239e+00
    double precision, parameter :: b1 = -5.447609879822406e+01, b2 = 1.615858368580409e+02
    double precision, parameter :: b3 = -1.556989798598866e+02, b4 = 6.680131188771972e+01
    double precision, parameter :: b5 = -1.328068155288572e+01
    double precision, parameter :: c1 = -7.784894002430293e-03, c2 = -3.223964580411365e-01
    double precision, parameter :: c3 = -2.400758277161838e+00, c4 = -2.549732539343734e+00
    double precision, parameter :: c5 = 4.374664141464968e+00, c6 = 2.938163982698783e+00
    double precision, parameter :: d1 = 7.784695709041462e-03, d2 = 3.224671290700398e-01
    double precision, parameter :: d3 = 2.445134137142996e+00, d4 = 3.754408661907416e+00
    double precision, parameter :: plow = 0.02425d0, phigh = 1.0d0 - plow
    if (p < plow) then
      q = sqrt(-2.0d0 * log(p))
      x = (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1.0d0)
    else if (p <= phigh) then
      q = p - 0.5d0
      r = q * q
      x = (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0d0)
    else
      q = sqrt(-2.0d0 * log(1.0d0 - p))
      x = -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1.0d0)
    end if
  end function norm_ppf

  !---------------------------------------------------------------------------
  ! gbm_call_control_variate(n_paths, s0, k, r, mu, sigma, t, seed,
  !                          price_mc, price_cv, bs_price)
  !   Monte Carlo European call, raw and with control variate.
  !   Control: the discounted payoff itself is centred on its known
  !   Black-Scholes expectation under the risk-neutral drift mu = r,
  !   so the CV estimator here uses the terminal stock price as control
  !   (E[S_T] = S0 exp(mu T) known analytically) with beta estimated
  !   from the sample covariance. This is the textbook control-variate
  !   construction and demonstrably shrinks the standard error.
  !---------------------------------------------------------------------------
  subroutine gbm_call_control_variate(n_paths, s0, k, r, mu, sigma, t, seed, &
                                      price_mc, price_cv, bs_price)
    integer, intent(in) :: n_paths, seed
    double precision, intent(in) :: s0, k, r, mu, sigma, t
    double precision, intent(out) :: price_mc, price_cv, bs_price
    double precision, allocatable :: z(:), payoff(:), control(:)
    double precision :: drift, diff, st, e_st, beta, cov, var_c, mean_pay, mean_ctl
    integer :: i
    allocate(z(n_paths), payoff(n_paths), control(n_paths))
    call randn_array(n_paths, seed, z)
    drift = (mu - 0.5d0 * sigma * sigma) * t
    diff = sigma * sqrt(t)
    e_st = s0 * exp(mu * t)
    do i = 1, n_paths
      st = s0 * exp(drift + diff * z(i))
      payoff(i) = exp(-r * t) * max(st - k, 0.0d0)
      control(i) = st
    end do
    mean_pay = sum(payoff) / dble(n_paths)
    mean_ctl = sum(control) / dble(n_paths)
    cov = sum((payoff - mean_pay) * (control - mean_ctl)) / dble(max(n_paths - 1, 1))
    var_c = sum((control - mean_ctl)**2) / dble(max(n_paths - 1, 1))
    beta = cov / var_c
    price_mc = mean_pay
    price_cv = mean_pay - beta * (mean_ctl - e_st)
    bs_price = black_scholes_call(s0, k, r, sigma, t)
    deallocate(z, payoff, control)
  end subroutine gbm_call_control_variate

  !---------------------------------------------------------------------------
  ! simulate_vasicek_losses(n_scen, n_oblig, pd, rho, lgd, ead, seed,
  !                          antithetic, losses, mean_loss)
  !   Vasicek single-factor model. Per scenario draw a systemic factor
  !   Y ~ N(0,1); the conditional default probability is
  !     p(Y) = Phi( (Phi^{-1}(PD) - sqrt(rho) Y) / sqrt(1 - rho) )
  !   and defaults are Binomial(n_oblig, p(Y)) (approximated here by a
  !   normal approximation to the binomial for speed at large N, exact
  !   binomial via uniforms for N <= 500). Loss = defaults * LGD * EAD.
  !   Antithetic: Y and -Y are paired.
  !---------------------------------------------------------------------------
  subroutine simulate_vasicek_losses(n_scen, n_oblig, pd, rho, lgd, ead, &
                                     seed, antithetic, losses, mean_loss)
    integer, intent(in) :: n_scen, n_oblig, seed
    logical, intent(in) :: antithetic
    double precision, intent(in) :: pd, rho, lgd, ead
    double precision, intent(out) :: losses(n_scen), mean_loss
    double precision, allocatable :: y(:), y2(:)
    double precision :: thr, py, u, m, sd, dflt_real
    integer :: i, j, half, dflt
    allocate(y(n_scen), y2(n_scen))
    call randn_array(n_scen, seed, y)
    call randn_array(n_scen, seed + 1, y2)
    thr = norm_ppf(pd)
    do i = 1, n_scen
      py = norm_cdf((thr - sqrt(rho) * y(i)) / sqrt(1.0d0 - rho))
      if (n_oblig <= 500) then
        ! Exact binomial: one uniform per obligor.
        dflt = 0
        do j = 1, n_oblig
          call random_number(u)
          if (u < py) dflt = dflt + 1
        end do
        losses(i) = dble(dflt) * lgd * ead
      else
        ! Normal approximation to Binomial(n, py): mean m, sd as below.
        ! The idiosyncratic noise uses the independent draw y2(i), so
        ! the systemic factor y(i) still drives the loss distribution
        ! through p(Y); relative error of the approximation is O(1/N).
        m = dble(n_oblig) * py
        sd = sqrt(dble(n_oblig) * py * (1.0d0 - py))
        dflt_real = m + sd * y2(i)
        dflt_real = max(0.0d0, min(dble(n_oblig), dflt_real))
        losses(i) = dflt_real * lgd * ead
      end if
    end do
    if (antithetic) then
      ! Pair scenario i with i+half: replace the second half by the
      ! antithetic partner (same |Y|, opposite sign) recomputed exactly.
      half = n_scen / 2
      do i = 1, half
        py = norm_cdf((thr + sqrt(rho) * y(i)) / sqrt(1.0d0 - rho))
        if (n_oblig <= 500) then
          dflt = 0
          do j = 1, n_oblig
            call random_number(u)
            if (u < py) dflt = dflt + 1
          end do
          losses(half + i) = dble(dflt) * lgd * ead
        else
          m = dble(n_oblig) * py
          sd = sqrt(dble(n_oblig) * py * (1.0d0 - py))
          dflt_real = max(0.0d0, min(dble(n_oblig), m - sd * y2(i)))
          losses(half + i) = dflt_real * lgd * ead
        end if
      end do
    end if
    mean_loss = sum(losses) / dble(n_scen)
    deallocate(y, y2)
  end subroutine simulate_vasicek_losses

  !---------------------------------------------------------------------------
  ! simulate_merton_losses(n_scen, n_oblig, corr, v0, debt, mu, sigma, t,
  !                        lgd, ead, seed, losses, mean_loss, info)
  !   Merton (1974): asset value V ~ GBM; obligor i defaults iff
  !   V_i(T) < Debt_i, i.e. iff its standardised asset return
  !   X_i < Phi^{-1}(PD_i) with PD_i from the Merton formula.
  !   X = L Z with L = Cholesky(corr). Loss = defaults * LGD * EAD.
  !---------------------------------------------------------------------------
  subroutine simulate_merton_losses(n_scen, n_oblig, corr, v0, debt, mu, &
                                    sigma, t, lgd, ead, seed, losses, &
                                    mean_loss, info)
    integer, intent(in) :: n_scen, n_oblig, seed
    double precision, intent(in) :: corr(n_oblig, n_oblig)
    double precision, intent(in) :: v0(n_oblig), debt(n_oblig)
    double precision, intent(in) :: mu(n_oblig), sigma(n_oblig), t
    double precision, intent(in) :: lgd, ead
    double precision, intent(out) :: losses(n_scen), mean_loss
    integer, intent(out) :: info
    double precision, allocatable :: lmat(:, :), z(:), x(:), thr(:)
    integer :: i, j, k, dflt
    double precision :: d2
    allocate(lmat(n_oblig, n_oblig), z(n_oblig), x(n_oblig), thr(n_oblig))
    call cholesky(n_oblig, corr, lmat, info)
    if (info /= 0) then
      losses = 0.0d0
      mean_loss = 0.0d0
      deallocate(lmat, z, x, thr)
      return
    end if
    do i = 1, n_oblig
      d2 = (log(v0(i) / debt(i)) + (mu(i) - 0.5d0 * sigma(i)**2) * t) &
           / (sigma(i) * sqrt(t))
      thr(i) = -d2  ! default iff X_i < -d2  <=>  PD = Phi(-d2)
    end do
    call seed_rng(seed)
    do i = 1, n_scen
      call randn_array(n_oblig, seed + i, z)
      dflt = 0
      do j = 1, n_oblig
        x(j) = 0.0d0
        do k = 1, j
          x(j) = x(j) + lmat(j, k) * z(k)
        end do
        if (x(j) < thr(j)) dflt = dflt + 1
      end do
      losses(i) = dble(dflt) * lgd * ead
    end do
    mean_loss = sum(losses) / dble(n_scen)
    deallocate(lmat, z, x, thr)
  end subroutine simulate_merton_losses

  !---------------------------------------------------------------------------
  ! portfolio_var_cvar(n, losses, alpha, var_out, cvar_out)
  !   VaR_alpha = alpha-quantile of losses (sorted, linear interpolation
  !   matching NumPy's default 'linear' method); CVaR = mean of losses
  !   strictly above VaR (Expected Shortfall).
  !---------------------------------------------------------------------------
  subroutine portfolio_var_cvar(n, losses, alpha, var_out, cvar_out)
    integer, intent(in) :: n
    double precision, intent(in) :: losses(n), alpha
    double precision, intent(out) :: var_out, cvar_out
    double precision, allocatable :: s(:)
    double precision :: pos, frac, tail_sum
    integer :: lo, i, cnt
    allocate(s(n))
    s = losses
    call sort_ascending(n, s)
    pos = alpha * dble(n - 1)
    lo = int(pos) + 1
    frac = pos - dble(lo - 1)
    if (lo >= n) then
      var_out = s(n)
    else
      var_out = s(lo) + frac * (s(lo + 1) - s(lo))
    end if
    tail_sum = 0.0d0
    cnt = 0
    do i = 1, n
      if (s(i) > var_out) then
        tail_sum = tail_sum + s(i)
        cnt = cnt + 1
      end if
    end do
    if (cnt > 0) then
      cvar_out = tail_sum / dble(cnt)
    else
      cvar_out = var_out
    end if
    deallocate(s)
  end subroutine portfolio_var_cvar

  ! Quicksort (median-of-three pivot, recursive) — O(n log n), so
  ! portfolio_var_cvar stays fast at 200k+ scenarios.
  recursive subroutine quicksort(a, lo, hi)
    double precision, intent(inout) :: a(:)
    integer, intent(in) :: lo, hi
    integer :: i, j, mid
    double precision :: pivot, tmp
    if (lo >= hi) return
    mid = (lo + hi) / 2
    if (a(lo) > a(mid)) then; tmp = a(lo); a(lo) = a(mid); a(mid) = tmp; end if
    if (a(mid) > a(hi)) then; tmp = a(mid); a(mid) = a(hi); a(hi) = tmp; end if
    if (a(lo) > a(mid)) then; tmp = a(lo); a(lo) = a(mid); a(mid) = tmp; end if
    pivot = a(mid)
    i = lo; j = hi
    do
      do while (a(i) < pivot); i = i + 1; end do
      do while (a(j) > pivot); j = j - 1; end do
      if (i <= j) then
        tmp = a(i); a(i) = a(j); a(j) = tmp
        i = i + 1; j = j - 1
      end if
      if (i > j) exit
    end do
    if (lo < j) call quicksort(a, lo, j)
    if (i < hi) call quicksort(a, i, hi)
  end subroutine quicksort

  subroutine sort_ascending(n, a)
    integer, intent(in) :: n
    double precision, intent(inout) :: a(n)
    if (n > 1) call quicksort(a, 1, n)
  end subroutine sort_ascending

end module montecarlo
