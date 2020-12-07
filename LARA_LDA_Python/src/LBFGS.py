import numpy as np

class LBFGS:
    def __init__(self):
        self.gtol = 2e-3
        self.stpmin = 1e-20
        self.stpmax = 1e20
        self.solution_cache = None
        self.gnorm = 0
        self.stp1 = 0
        self.ftol =0
        self.stp = None
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
    def lbfgs(self, n, m, x, f, g, diagco, diag, iprint, eps, xtol, iflag):
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
            print("LBFGS.lbfgs: gtol is less than or equal to 0.0001. It has been reset to 0.9.")
            self.gtol = 0.9

        self.nfun = 1
        self.point = 0
        self.finish = False

        # diagco = False
        for i in range(1, n+1):
			self.diag[i -1] = 1

        self.ispt = n+2*m
        self.iypt = self.ispt + n*m

        for i in range(1, n+1):
            self.w[self.ispt + 1 - 1] = -g[i-1] * self.diag[i-1]

        self.gnorm = np.sqrt(self.ddot(n, g, 0, 1, g, 0, 1))
        self.stp1 = 1 / self.gnorm
        self.ftol = 0.0001
        self.maxfev = 100

        if iprint[1 -1] >= 0:
            return

        return


    def ddot(self, n, dx, ix0, incx, dy, iy0, incy):
        if n <= 0:
            return 0

        dtemp = 0
        if not (incx == 1 and incy == 1):
            ix = 1
            iy = 1
            if incx < 0:
                ix = ( - n + 1 ) * incx + 1
            if incy < 0:
                iy = ( - n + 1 ) * incy + 1
            for i in range(1, n+1):
                dtemp = dtemp + dx[ix0+ix -1] * dy[iy0+iy -1];
                ix = ix + incx
                iy = iy + incy

            return dtemp

        m = n % 5
        if m != 0:
            for i in range(1, m+1):
                dtemp = dtemp + dx[ ix0+i -1] * dy [ iy0+i -1]
            if n < 5:
                return dtemp

        mp1 = m + 1
        for i in range(mp1, n+1, step=5):
            dtemp = dtemp + dx[ix0 + i - 1] * dy[iy0 + i - 1] + dx[ix0 + i + 1 - 1] * dy[iy0 + i + 1 - 1] + \
                    dx[ix0 + i + 2 - 1] * dy[iy0 + i + 2 - 1] + dx[ix0 + i + 3 - 1] * dy[iy0 + i + 3 - 1] + \
                    dx[ix0 + i + 4 - 1] * dy[iy0 + i + 4 - 1]

        return dtemp

    def lb1(self, iprint, iter, nfun, gnorm, n, m, x, f, g, stp, finish):
        return
