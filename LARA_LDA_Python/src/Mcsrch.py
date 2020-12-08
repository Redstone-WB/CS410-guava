import numpy as np

from LBFGS import *


class Mcsrch:
    @staticmethod
    def sqr(x):
        return x*x

    @staticmethod
    def max3(x, y, z):
        return np.max([x, y, z])

    @staticmethod
    def mcsrch(n, x, f, g, s, is0, stp, ftol, xtol, maxfev, info, nfev, wa):
        infoc = [0]
        j = 0
        dg = 0
        dgm = 0
        dginit = 0
        dgx = [0.0]
        dgxm = [0.0]
        dgy = [0.0]
        dgym = [0.0]
        finit = 0
        ftest1 = 0
        fm = 0
        fx = [0.0]
        fxm = [0.0]
        fy = [0.0]
        fym = [0.0]
        p5 = 0
        p66 = 0
        stx = [0.0]
        sty = [0.0]
        stmin = 0
        stmax = 0
        width = 0
        width1 = 0
        xtrapf = 0
        brackt = [False]
        stage1 = False

        p5 = 0.5
        p66 = 0.66
        xtrapf = 4

        if info[0] != - 1:
            infoc[0] = 1
            if n <= 0 or stp[0] <= 0 or ftol < 0 or LBFGS.gtol < 0 or xtol < 0 or LBFGS.stpmin < 0 or LBFGS.stpmax < LBFGS.stpmin or maxfev <= 0:
                return

            # Compute the initial gradient in the search direction
            # and check that s is a descent direction.

            dginit = 0

            for j in range(1, n+1):
                dginit = dginit + g[j - 1] * s[is0 + j - 1]

            if dginit >= 0:
                print("The search direction is not a descent direction.")
                return

            brackt[0] = False
            stage1 = True
            nfev[0] = 0
            finit = f
            dgtest = ftol * dginit
            width = LBFGS.stpmax - LBFGS.stpmin
            width1 = width / p5

            for j in range(1, n+1):
                wa[j - 1] = x[j - 1]

            # The variables stx, fx, dgx contain the values of the step,
            # function, and directional derivative at the best step.
            # The variables sty, fy, dgy contain the value of the step,
            # function, and derivative at the other endpoint of
            # the interval of uncertainty.
            # The variables stp, f, dg contain the values of the step,
            # function, and derivative at the current step.

            stx[0] = 0
            fx[0] = finit
            dgx[0] = dginit
            sty[0] = 0
            fy[0] = finit
            dgy[0] = dginit

        while True:
            if info[0] != -1:
                # Set the minimum and maximum steps to correspond
                # to the present interval of uncertainty.

                if brackt[0]:
                    stmin = np.min([stx[0], sty[0]])
                    stmax = np.max([stx[0], sty[0]])
                else:
                    stmin = stx[0]
                    stmax = stp[0] + xtrapf * (stp[0] - stx[0])

                # Force the step to be within the bounds stpmax and stpmin.
                stp[0] = np.max([stp[0], LBFGS.stpmin])
                stp[0] = np.min([stp[0], LBFGS.stpmax])

                # If an unusual termination is to occur then let
                # stp be the lowest point obtained so far.

                if (brackt[0] and (stp[0] <= stmin or stp[0] >= stmax)) or nfev[0] >= maxfev - 1 or infoc[0] == 0 or (brackt[0] and stmax - stmin <= xtol * stmax):
                    stp[0] = stx[0]

                # Evaluate the function and gradient at stp
                # and compute the directional derivative.
                # We return to main program to obtain F and G.

                for j in range(1, n+1):
                    x[j - 1] = wa[j - 1] + stp[0] * s[is0 + j - 1]

                info[0] = -1
                return

            info[0] = 0
            nfev[0] = nfev[0] + 1
            dg = 0

            for j in range(1, n+1):
                dg = dg + g[j - 1] * s[is0 + j - 1]

            ftest1 = finit + stp[0] * dgtest

            # Test for convergence.

            if (brackt[0] and (stp[0] <= stmin or stp[0] >= stmax)) or infoc[0] == 0:
                info[0] = 6

            if stp[0] == LBFGS.stpmax and f <= ftest1 and dg <= dgtest:
                info[0] = 5

            if stp[0] == LBFGS.stpmin and ( f > ftest1 or dg >= dgtest ):
                info[0] = 4

            if nfev[0] >= maxfev:
                info[0] = 3

            if brackt[0] and stmax - stmin <= xtol * stmax:
                info[0] = 2

            if f <= ftest1 and np.abs(dg) <= LBFGS.gtol * ( - dginit ):
                info[0] = 1

            # Check for termination.

            if info[0] != 0:
                return

            # In the first stage we seek a step for which the modified
            # function has a nonpositive value and nonnegative derivative.

            if stage1 and f <= ftest1 and dg >= np.min([ftol, LBFGS.gtol]) * dginit:
                stage1 = False

            # A modified function is used to predict the step only if
            # we have not obtained a step for which the modified
            # function has a nonpositive function value and nonnegative
            # derivative, and if a lower function value has been
            # obtained but the decrease is not sufficient.

            if stage1 and f <= fx[0] and f > ftest1:
                # Define the modified function and derivative values.

                fm = f - stp[0] * dgtest
                fxm[0] = fx[0] - stx[0] * dgtest
                fym[0] = fy[0] - sty[0] * dgtest
                dgm = dg - dgtest
                dgxm[0] = dgx[0] - dgtest
                dgym[0] = dgy[0] - dgtest

                # Call cstep to update the interval of uncertainty
                # and to compute the new step.
                Mcsrch.mcstep(stx, fxm, dgxm, sty, fym, dgym, stp, fm, dgm, brackt, stmin, stmax, infoc)

                # Reset the function and gradient values for f.

                fx[0] = fxm[0] + stx[0] * dgtest
                fy[0] = fym[0] + sty[0] * dgtest
                dgx[0] = dgxm[0] + dgtest
                dgy[0] = dgym[0] + dgtest
            else:
                # Call mcstep to update the interval of uncertainty
                # and to compute the new step.
                Mcsrch.mcstep(stx, fx, dgx, sty, fy, dgy, stp, f, dg, brackt, stmin, stmax, infoc)

            # Force a sufficient decrease in the size of the
            # interval of uncertainty.

            if brackt[0]:
                if np.abs(sty[0] - stx[0]) >= p66 * width1:
                    stp[0] = stx[0] + p5 * (sty[0] - stx[0])
                width1 = width
                width = np.abs(sty[0] - stx[0])

    @staticmethod
    def mcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax, info):
        info[0] = 0

        if (brackt[0] and (stp[0] <= np.min([stx[0], sty[0]]) or stp[0] >= np.max([stx[0], sty[0]]))) or dx[0] * (stp[0] - stx[0]) >= 0.0 or stpmax < stpmin:
            return

        sgnd = dp * (dx[0] / np.abs(dx[0]))

        if fp > fx[0]:
            # First case
            info[0] = 1
            bound = True
            theta = 3 * (fx[0] - fp) / (stp[0] - stx[0]) + dx[0] + dp
            s = Mcsrch.max3(np.abs(theta), np.abs(dx[0]), np.abs(dp))
            gamma = s * np.sqrt(Mcsrch.sqr(theta / s) - (dx[0] / s) * (dp / s))
            if stp[0] < stx[0]:
                gamma = - gamma
            p = (gamma - dx[0]) + theta
            q = ((gamma - dx[0]) + gamma) + dp
            r = p / q
            stpc = stx[0] + r * (stp[0] - stx[0])
            stpq = stx[0] + ((dx[0] / ((fx[0] - fp) / (stp[0] - stx[0]) + dx[0])) / 2) * (stp[0] - stx[0])
            if np.abs(stpc - stx[0]) < np.abs(stpq - stx[0]):
                stpf = stpc
            else:
                stpf = stpc + (stpq - stpc) / 2
            brackt[0] = True
        elif sgnd < 0.0:
            # Second case
            info[0] = 2
            bound = False
            theta = 3 * (fx[0] - fp) / (stp[0] - stx[0]) + dx[0] + dp
            s = Mcsrch.max3(np.abs(theta), np.abs(dx[0]), np.abs(dp))
            gamma = s * np.sqrt(Mcsrch.sqr(theta / s) - (dx[0] / s) * (dp / s))
            if stp[0] > stx[0]:
                gamma = - gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dx[0]
            r = p / q
            stpc = stp[0] + r * (stx[0] - stp[0])
            stpq = stp[0] + (dp / (dp - dx[0])) * (stx[0] - stp[0])
            if np.abs(stpc - stp[0]) > np.abs(stpq - stp[0]):
                stpf = stpc
            else:
                stpf = stpq
            brackt[0] = True
        elif np.abs(dp) < np.abs (dx[0]):
            # Third case
            info[0] = 3
            bound = True
            theta = 3 * (fx[0] - fp) / (stp[0] - stx[0]) + dx[0] + dp
            s = Mcsrch.max3(np.abs(theta), np.abs(dx[0]), np.abs(dp))
            gamma = s * np.sqrt(np.max([0, Mcsrch.sqr(theta / s) - (dx[0] / s) * (dp / s)]));
            if stp[0] > stx[0]:
                gamma = - gamma;
            p = (gamma - dp) + theta
            q = (gamma + (dx[0] - dp)) + gamma
            r = p / q
            if r < 0.0 and gamma != 0.0:
                stpc = stp[0] + r * (stx[0] - stp[0])
            elif stp[0] > stx[0]:
                stpc = stpmax
            else:
                stpc = stpmin

            stpq = stp[0] + (dp / (dp - dx[0])) * (stx[0] - stp[0])
            if brackt[0]:
                if np.abs(stp[0] - stpc) < np.abs(stp[0] - stpq):
                    stpf = stpc
                else:
                    stpf = stpq
            else:
                if np.abs(stp[0] - stpc) > np.abs(stp[0] - stpq):
                    stpf = stpc
                else:
                    stpf = stpq
        else:
            # Fourth case.
            info[0] = 4
            bound = False
            if brackt[0]:
                theta = 3 * (fp - fy[0]) / (sty[0] - stp[0]) + dy[0] + dp
                s = Mcsrch.max3(np.abs(theta), np.abs(dy[0]), np.abs(dp))
                gamma = s * np.sqrt(Mcsrch.sqr(theta / s) - (dy[0] / s) * (dp / s))
                if stp[0] > sty[0]:
                    gamma = - gamma
                p = ( gamma - dp ) + theta
                q = ( ( gamma - dp ) + gamma ) + dy[0]
                r = p / q
                stpc = stp[0] + r * ( sty[0] - stp[0] )
                stpf = stpc
            elif stp[0] > stx[0]:
                stpf = stpmax
            else:
                stpf = stpmin

        # Update the interval of uncertainty. This update does not
        # depend on the new step or the case analysis above.

        if fp > fx[0]:
            sty[0] = stp[0]
            fy[0] = fp
            dy[0] = dp
        else:
            if sgnd < 0.0:
                sty[0] = stx[0]
                fy[0] = fx[0]
                dy[0] = dx[0]
            stx[0] = stp[0]
            fx[0] = fp
            dx[0] = dp

        # Compute the new step and safeguard it.

        stpf = np.min([stpmax, stpf])
        stpf = np.max([stpmin, stpf])
        stp[0] = stpf

        if brackt[0] and bound:
            if sty[0] > stx[0]:
                stp[0] = np.min([stx[0] + 0.66 * (sty[0] - stx[0]), stp[0]])
            else:
                stp[0] = np.max([stx[0] + 0.66 * (sty[0] - stx[0]), stp[0]])

        return

