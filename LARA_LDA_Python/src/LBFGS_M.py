import numpy as np

from src.ExceptionWithIflag import ExceptionWithIflag
from src.Mcsrch import Mcsrch


class LBFGS:

    def __init__(self):
        self.gtol = 2e-3
        self.stpmin = 1e-20
        self.stpmax = 1e20
        self.solution_cache = None
        self.gnorm = 0
        self.stp1 = 0
        self.ftol = 0
        self.stp = [0.0]
        self.ys = 0
        self.yy = 0
        self.sq = 0
        self.yr = 0
        self.beta = 0
        self.xnorm = 0
        self.iter = 0
        self.nfun = 0
        self.point = 0
        self.ispt = 0
        self.iypt = 0
        self.maxfev = 0
        self.info = [0]
        self.bound = 0
        self.npt = 0
        self.cp = 0
        self.i = 0
        self.nfev = [0]
        self.inmc = 0
        self.iycn = 0
        self.iscn = 0
        self.finish = False
        self.w = None

    # public static void lbfgs ( int n , int m , double[] x , double f , double[] g , boolean diagco , double[] diag , int[] iprint , double eps , double xtol , int[] iflag ) throws ExceptionWithIflag

    def lbfgs_func(self, n, m, x, f, g, diagco, diag, iprint, eps, xtol, iflag):
        execute_entire_while_loop = False

        if self.w is None or len(self.w) != n * (2 * m + 1) + 2 * m:
            self.w = np.zeros(n * (2 * m + 1) + 2 * m)

        if iflag[0] == 0:
            # Initialize.
            self.solution_cache = x.copy()

            iter = 0

            if n <= 0 or m <= 0:
                print("Improper input parameters  (n or m are not positive.)")
                iflag[0] = -3
                return

            if self.gtol <= 0.0001:
                print(
                    "LBFGS.lbfgs: gtol is less than or equal to 0.0001. It has been reset to 0.9.")
                self.gtol = 0.9

            self.nfun = 1
            self.point = 0
            self.finish = False

            # diagco = False
            for i in range(1, n+1):
                diag[i - 1] = 1

            self.ispt = n+2*m
            self.iypt = self.ispt + n*m

            for i in range(1, n+1):
                self.w[self.ispt + i - 1] = -g[i-1] * diag[i-1]

            ttemp = self.ddot(n, g, 0, 1, g, 0, 1)
            self.gnorm = np.sqrt(ttemp)
            self.stp1 = 1 / self.gnorm
            self.ftol = 0.0001
            self.maxfev = 100

            if iprint[1-1] >= 0:
                self.lb1(iprint, iter, self.nfun, self.gnorm,
                         n, m, x, f, g, self.stp, self.finish)

            execute_entire_while_loop = True

        while True:
            if execute_entire_while_loop:
                self.iter += 1
                self.info[0] = 0
                self.bound = self.iter-1
                if self.iter != 1:
                    if self.iter > m:
                        self.bound = m
                    self.ys = self.ddot(
                        n, self.w, self.iypt + self.npt, 1, self.w, self.ispt + self.npt, 1)
                    # diagco = False
                    self.yy = self.ddot(
                        n, self.w, self.iypt + self.npt, 1, self.w, self.iypt + self.npt, 1)
                    for i in range(1, n+1):
                        diag[i-1] = self.ys / self.yy

            if execute_entire_while_loop or iflag[0] == 2:
                if self.iter != 1:
                    # diagco = False
                    self.cp = self.point
                    if self.point == 0:
                        self.cp = m
                    self.w[n + self.cp-1] = 1 / self.ys

                    for i in range(1, n+1):
                        self.w[i-1] = -g[i-1]

                    self.cp = self.point

                    for i in range(1, self.bound+1):
                        self.cp = self.cp - 1
                        if self.cp == -1:
                            self.cp = m - 1
                        self.sq = self.ddot(
                            n, self.w, self.ispt + self.cp * n, 1, self.w, 0, 1)
                        self.inmc = n+m+self.cp+1
                        self.iycn = self.iypt + self.cp * n
                        self.w[self.inmc - 1] = self.w[n +
                                                       self.cp + 1 - 1] * self.sq
                        self.daxpy(n, - self.w[self.inmc - 1],
                                   self.w, self.iycn, 1, self.w, 0, 1)

                    for i in range(1, n+1):
                        self.w[i - 1] = diag[i - 1] * self.w[i - 1]

                    for i in range(1, self.bound+1):
                        self.yr = self.ddot(
                            n, self.w, self.iypt + self.cp * n, 1, self.w, 0, 1)
                        self.beta = self.w[n + self.cp + 1 - 1] * self.yr
                        self.inmc = n + m + self.cp + 1
                        self.beta = self.w[self.inmc - 1] - self.beta
                        self.iscn = self.ispt + self.cp * n
                        self.daxpy(n, self.beta, self.w,
                                   self.iscn, 1, self.w, 0, 1)
                        self.cp += 1
                        if self.cp == m:
                            self.cp = 0

                    for i in range(1, n+1):
                        self.w[self.ispt + self.point *
                               n + i - 1] = self.w[i - 1]

                self.nfev[0] = 0
                self.stp[0] = 1
                if self.iter == 1:
                    self.stp[0] = self.stp1

                for i in range(1, n+1):
                    self.w[i - 1] = g[i - 1]

            Mcsrch.mcsrch(n, x, f, g, self.w, self.ispt + self.point * n, self.stp, self.ftol, xtol,
                          self.maxfev, self.info, self.nfev, diag, self.gtol, self.stpmin, self.stpmax)

            if self.info[0] == -1:
                iflag[0] = 1
                return

            if self.info[0] != 1:
                iflag[0] = -1
                # print("Line search failed. See documentation of routine mcsrch. Error return of line search")
                return -1

            self.nfun = self.nfun + self.nfev[0]
            self.npt = self.point * n

            for i in range(1, n+1):
                self.w[self.ispt + self.npt + i - 1] = self.stp[0] * \
                    self.w[self.ispt + self.npt + i - 1]
                self.w[self.iypt + self.npt + i - 1] = g[i - 1] - self.w[i - 1]

            self.point = self.point + 1
            if self.point == m:
                self.point = 0

            self.gnorm = np.sqrt(self.ddot(n, g, 0, 1, g, 0, 1))
            self.xnorm = np.sqrt(self.ddot(n, x, 0, 1, x, 0, 1))
            self.xnorm = np.max([1.0, self.xnorm])

            if self.gnorm / self.xnorm <= eps:
                self.finish = True

            if iprint[1 - 1] >= 0:
                self.lb1(iprint, iter, self.nfun, self.gnorm,
                         n, m, x, f, g, self.stp, self.finish)

            # Cache the current solution vector. Due to the spaghetti-like
            # nature of this code, it's not possible to quit here and return;
            # we need to go back to the top of the loop, and eventually call
            # mcsrch one more time -- but that will modify the solution vector.
            # So we need to keep a copy of the solution vector as it was at
            # the completion (info[0]==1) of the most recent line search.

            self.solution_cache = x.copy()

            if self.finish:
                iflag[0] = 0
                return

            execute_entire_while_loop = True  # from now on, execute whole loop

    def ddot(self, n, dx, ix0, incx, dy, iy0, incy):
        if n <= 0:
            return 0

        dtemp = 0
        if not (incx == 1 and incy == 1):
            ix = 1
            iy = 1
            if incx < 0:
                ix = (- n + 1) * incx + 1
            if incy < 0:
                iy = (- n + 1) * incy + 1
            for i in range(1, n+1):
                dtemp = dtemp + dx[ix0+ix - 1] * dy[iy0+iy - 1]
                ix = ix + incx
                iy = iy + incy

            return dtemp

        m = n % 5
        if m != 0:
            for i in range(1, m+1):
                dtemp = dtemp + dx[ix0+i - 1] * dy[iy0+i - 1]
            if n < 5:
                return dtemp

        mp1 = m + 1
        for i in range(mp1, n+1, 5):
            dtemp = dtemp + dx[ix0 + i - 1] * dy[iy0 + i - 1] + dx[ix0 + i + 1 - 1] * dy[iy0 + i + 1 - 1] + \
                dx[ix0 + i + 2 - 1] * dy[iy0 + i + 2 - 1] + dx[ix0 + i + 3 - 1] * dy[iy0 + i + 3 - 1] + \
                dx[ix0 + i + 4 - 1] * dy[iy0 + i + 4 - 1]
            # print("{}: dtemp={}".format(i, dtemp))

        return dtemp

    def lb1(self, iprint, iter, nfun, gnorm, n, m, x, f, g, stp, finish):
        return

    def daxpy(self, n, da, dx, ix0, incx, dy, iy0, incy):
        if n <= 0:
            return
        if da == 0:
            return

        if not (incx == 1 and incy == 1):
            ix = 1
            iy = 1

            if incx < 0:
                ix = (-n + 1) * self.incx + 1
            if incy < 0:
                iy = (-n + 1) * self.incy + 1

            for i in range(1, n+1):
                dy[iy0+iy - 1] = dy[iy0+iy - 1] + da * dx[ix0+ix - 1]
                ix = ix + incx
                iy = iy + incy
            return

        m = n % 4
        if m != 0:
            for i in range(1, m+1):
                dy[iy0 + i - 1] = dy[iy0+i - 1] + da * dx[ix0+i - 1]

            if n < 4:
                return

        mp1 = m + 1
        for i in range(mp1, n+1, 4):
            dy[iy0 + i - 1] = dy[iy0 + i - 1] + da * dx[ix0 + i - 1]
            dy[iy0 + i + 1 - 1] = dy[iy0 + i + 1 - 1] + da * dx[ix0 + i + 1 - 1]
            dy[iy0 + i + 2 - 1] = dy[iy0 + i + 2 - 1] + da * dx[ix0 + i + 2 - 1]
            dy[iy0 + i + 3 - 1] = dy[iy0 + i + 3 - 1] + da * dx[ix0 + i + 3 - 1]


if __name__ == "__main__":
    optimizer = LBFGS()
    m = 5
    f = 1081.9831533640386

    with open('/workspace/beta.txt', 'r') as x_file, \
            open('/workspace/g_beta.txt', 'r') as g_file:
        m_alpha_hat = []
        x_data = x_file.read()
        for item in x_data.split(","):
            if item == '':
                continue
            m_alpha_hat.append(item)
        m_alpha_hat = np.array(m_alpha_hat, dtype=np.float64)

        m_g_alpha = []
        g_data = g_file.read()
        for item in g_data.split(","):
            if item == '':
                continue
            m_g_alpha.append(item)
        m_g_alpha = np.array(m_g_alpha, dtype=np.float64)

        m_diag_alpha = np.zeros(len(m_alpha_hat))

        n = len(m_alpha_hat)
        iprint = [-1, 0]
        m_alphaTol = 0.01
        iflag = [0]

        ret = optimizer.lbfgs_func(
            n, m, m_alpha_hat, f, m_g_alpha, False,
            m_diag_alpha, iprint, m_alphaTol, 1e-20, iflag)

        AssertionError(m_alpha_hat[0] == -0.7075550708792452)
        AssertionError(m_g_alpha[0] == -1.313301233992803)
        AssertionError(m_diag_alpha[0] == -1.0179588329040197)
        AssertionError(iflag[0] == 1)
