import numpy as np


class Mcsrch():
    infoc = [0]
    j = 0
    dg = 0
    dgm = 0
    dginit = 0
    dgtest = 0
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

    @staticmethod
    def sqr(x):
        return x*x

    @staticmethod
    def max3(x, y, z):
        return np.max([x, y, z])

    @staticmethod
    def mcsrch(n, x, f, g, s, is0, stp, ftol, xtol, maxfev, info, nfev, wa, gtol, stpmin, stpmax):
        p5 = 0.5
        p66 = 0.66
        xtrapf = 4

        if info[0] != - 1:
            Mcsrch.infoc[0] = 1
            if n <= 0 or stp[0] <= 0 or ftol < 0 or gtol < 0 or xtol < 0 or stpmin < 0 or stpmax < stpmin or maxfev <= 0:
                return

            # Compute the initial gradient in the search direction
            # and check that s is a descent direction.

            Mcsrch.dginit = 0

            for j in range(1, n+1):
                Mcsrch.dginit = Mcsrch.dginit + g[j - 1] * s[is0 + j - 1]

            if Mcsrch.dginit >= 0:
                print("The search direction is not a descent direction.")
                return

            Mcsrch.brackt[0] = False
            Mcsrch.stage1 = True
            nfev[0] = 0
            Mcsrch.finit = f
            Mcsrch.dgtest = ftol * Mcsrch.dginit
            Mcsrch.width = stpmax - stpmin
            Mcsrch.width1 = Mcsrch.width / p5

            for j in range(1, n+1):
                wa[j - 1] = x[j - 1]

            # The variables stx, fx, dgx contain the values of the step,
            # function, and directional derivative at the best step.
            # The variables sty, fy, dgy contain the value of the step,
            # function, and derivative at the other endpoint of
            # the interval of uncertainty.
            # The variables stp, f, dg contain the values of the step,
            # function, and derivative at the current step.

            Mcsrch.stx[0] = 0
            Mcsrch.fx[0] = Mcsrch.finit
            Mcsrch.dgx[0] = Mcsrch.dginit
            Mcsrch.sty[0] = 0
            Mcsrch.fy[0] = Mcsrch.finit
            Mcsrch.dgy[0] = Mcsrch.dginit

        while True:
            if info[0] != -1:
                # Set the minimum and maximum steps to correspond
                # to the present interval of uncertainty.

                if Mcsrch.brackt[0]:
                    Mcsrch.stmin = np.min([Mcsrch.stx[0], Mcsrch.sty[0]])
                    Mcsrch.stmax = np.max([Mcsrch.stx[0], Mcsrch.sty[0]])
                else:
                    Mcsrch.stmin = Mcsrch.stx[0]
                    Mcsrch.stmax = stp[0] + xtrapf * (stp[0] - Mcsrch.stx[0])

                # Force the step to be within the bounds stpmax and stpmin.
                stp[0] = np.max([stp[0], stpmin])
                stp[0] = np.min([stp[0], stpmax])

                # If an unusual termination is to occur then let
                # stp be the lowest point obtained so far.

                if (Mcsrch.brackt[0] and (stp[0] <= Mcsrch.stmin or stp[0] >= Mcsrch.stmax)) or nfev[0] >= maxfev - 1 or Mcsrch.infoc[0] == 0 or (Mcsrch.brackt[0] and Mcsrch.stmax - Mcsrch.stmin <= xtol * Mcsrch.stmax):
                    stp[0] = Mcsrch.stx[0]

                # Evaluate the function and gradient at stp
                # and compute the directional derivative.
                # We return to main program to obtain F and G.

                for j in range(1, n+1):
                    x[j - 1] = wa[j - 1] + stp[0] * s[is0 + j - 1]

                info[0] = -1
                return

            info[0] = 0
            nfev[0] = nfev[0] + 1
            Mcsrch.dg = 0

            for j in range(1, n+1):
                Mcsrch.dg = Mcsrch.dg + g[j - 1] * s[is0 + j - 1]

            ftest1 = Mcsrch.finit + stp[0] * Mcsrch.dgtest

            # Test for convergence.

            if (Mcsrch.brackt[0] and (stp[0] <= Mcsrch.stmin or stp[0] >= Mcsrch.stmax)) or Mcsrch.infoc[0] == 0:
                info[0] = 6

            if stp[0] == stpmax and f <= ftest1 and Mcsrch.dg <= Mcsrch.dgtest:
                info[0] = 5

            if stp[0] == stpmin and (f > ftest1 or Mcsrch.dg >= Mcsrch.dgtest):
                info[0] = 4

            if nfev[0] >= maxfev:
                info[0] = 3

            if Mcsrch.brackt[0] and Mcsrch.stmax - Mcsrch.stmin <= xtol * Mcsrch.stmax:
                info[0] = 2

            if f <= ftest1 and np.abs(Mcsrch.dg) <= gtol * (- Mcsrch.dginit):
                info[0] = 1

            # Check for termination.

            if info[0] != 0:
                return

            # In the first stage we seek a step for which the modified
            # function has a nonpositive value and nonnegative derivative.

            if Mcsrch.stage1 and f <= ftest1 and Mcsrch.dg >= np.min([ftol, gtol]) * Mcsrch.dginit:
                Mcsrch.stage1 = False

            # A modified function is used to predict the step only if
            # we have not obtained a step for which the modified
            # function has a nonpositive function value and nonnegative
            # derivative, and if a lower function value has been
            # obtained but the decrease is not sufficient.

            if Mcsrch.stage1 and f <= Mcsrch.fx[0] and f > ftest1:
                # Define the modified function and derivative values.

                Mcsrch.fm = f - stp[0] * Mcsrch.dgtest
                Mcsrch.fxm[0] = Mcsrch.fx[0] - Mcsrch.stx[0] * Mcsrch.dgtest
                Mcsrch.fym[0] = Mcsrch.fy[0] - Mcsrch.sty[0] * Mcsrch.dgtest
                Mcsrch.dgm = Mcsrch.dg - Mcsrch.dgtest
                Mcsrch.dgxm[0] = Mcsrch.dgx[0] - Mcsrch.dgtest
                Mcsrch.dgym[0] = Mcsrch.dgy[0] - Mcsrch.dgtest

                # Call cstep to update the interval of uncertainty
                # and to compute the new step.
                Mcsrch.mcstep(Mcsrch.stx, Mcsrch.fxm, Mcsrch.dgxm, Mcsrch.sty, Mcsrch.fym, Mcsrch.dgym,
                              stp, Mcsrch.fm, Mcsrch.dgm, Mcsrch.brackt, Mcsrch.stmin, Mcsrch.stmax, Mcsrch.infoc)

                # Reset the function and gradient values for f.

                Mcsrch.fx[0] = Mcsrch.fxm[0] + Mcsrch.stx[0] * Mcsrch.dgtest
                Mcsrch.fy[0] = Mcsrch.fym[0] + Mcsrch.sty[0] * Mcsrch.dgtest
                Mcsrch.dgx[0] = Mcsrch.dgxm[0] + Mcsrch.dgtest
                Mcsrch.dgy[0] = Mcsrch.dgym[0] + Mcsrch.dgtest
            else:
                # Call mcstep to update the interval of uncertainty
                # and to compute the new step.
                Mcsrch.mcstep(Mcsrch.stx, Mcsrch.fx, Mcsrch.dgx, Mcsrch.sty, Mcsrch.fy, Mcsrch.dgy,
                              stp, f, Mcsrch.dg, Mcsrch.brackt, Mcsrch.stmin, Mcsrch.stmax, Mcsrch.infoc)

            # Force a sufficient decrease in the size of the
            # interval of uncertainty.

            if Mcsrch.brackt[0]:
                if np.abs(Mcsrch.sty[0] - Mcsrch.stx[0]) >= p66 * Mcsrch.width1:
                    stp[0] = Mcsrch.stx[0] + p5 * \
                        (Mcsrch.sty[0] - Mcsrch.stx[0])
                Mcsrch.width1 = Mcsrch.width
                Mcsrch.width = np.abs(Mcsrch.sty[0] - Mcsrch.stx[0])

    @staticmethod
    def mcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax, info):
        info[0] = 0

        if (Mcsrch.brackt[0] and (stp[0] <= np.min([Mcsrch.stx[0], Mcsrch.sty[0]]) or stp[0] >= np.max([Mcsrch.stx[0], Mcsrch.sty[0]]))) or dx[0] * (stp[0] - Mcsrch.stx[0]) >= 0.0 or stpmax < stpmin:
            return

        sgnd = dp * (dx[0] / np.abs(dx[0]))

        if fp > Mcsrch.fx[0]:
            # First case
            info[0] = 1
            bound = True
            theta = 3 * (Mcsrch.fx[0] - fp) / \
                (stp[0] - Mcsrch.stx[0]) + dx[0] + dp
            s = Mcsrch.max3(np.abs(theta), np.abs(dx[0]), np.abs(dp))
            gamma = s * np.sqrt(Mcsrch.sqr(theta / s) - (dx[0] / s) * (dp / s))
            if stp[0] < Mcsrch.stx[0]:
                gamma = - gamma
            p = (gamma - dx[0]) + theta
            q = ((gamma - dx[0]) + gamma) + dp
            r = p / q
            stpc = Mcsrch.stx[0] + r * (stp[0] - Mcsrch.stx[0])
            stpq = Mcsrch.stx[0] + ((dx[0] / ((Mcsrch.fx[0] - fp) / (
                stp[0] - Mcsrch.stx[0]) + dx[0])) / 2) * (stp[0] - Mcsrch.stx[0])
            if np.abs(stpc - Mcsrch.stx[0]) < np.abs(stpq - Mcsrch.stx[0]):
                stpf = stpc
            else:
                stpf = stpc + (stpq - stpc) / 2
            Mcsrch.brackt[0] = True
        elif sgnd < 0.0:
            # Second case
            info[0] = 2
            bound = False
            theta = 3 * (Mcsrch.fx[0] - fp) / \
                (stp[0] - Mcsrch.stx[0]) + dx[0] + dp
            s = Mcsrch.max3(np.abs(theta), np.abs(dx[0]), np.abs(dp))
            gamma = s * np.sqrt(Mcsrch.sqr(theta / s) - (dx[0] / s) * (dp / s))
            if stp[0] > Mcsrch.stx[0]:
                gamma = - gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dx[0]
            r = p / q
            stpc = stp[0] + r * (Mcsrch.stx[0] - stp[0])
            stpq = stp[0] + (dp / (dp - dx[0])) * (Mcsrch.stx[0] - stp[0])
            if np.abs(stpc - stp[0]) > np.abs(stpq - stp[0]):
                stpf = stpc
            else:
                stpf = stpq
            Mcsrch.brackt[0] = True
        elif np.abs(dp) < np.abs(dx[0]):
            # Third case
            info[0] = 3
            bound = True
            theta = 3 * (Mcsrch.fx[0] - fp) / \
                (stp[0] - Mcsrch.stx[0]) + dx[0] + dp
            s = Mcsrch.max3(np.abs(theta), np.abs(dx[0]), np.abs(dp))
            gamma = s * \
                np.sqrt(
                    np.max([0, Mcsrch.sqr(theta / s) - (dx[0] / s) * (dp / s)]))
            if stp[0] > Mcsrch.stx[0]:
                gamma = - gamma
            p = (gamma - dp) + theta
            q = (gamma + (dx[0] - dp)) + gamma
            r = p / q
            if r < 0.0 and gamma != 0.0:
                stpc = stp[0] + r * (Mcsrch.stx[0] - stp[0])
            elif stp[0] > Mcsrch.stx[0]:
                stpc = stpmax
            else:
                stpc = stpmin

            stpq = stp[0] + (dp / (dp - dx[0])) * (Mcsrch.stx[0] - stp[0])
            if Mcsrch.brackt[0]:
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
            if Mcsrch.brackt[0]:
                theta = 3 * (fp - Mcsrch.fy[0]) / \
                    (Mcsrch.sty[0] - stp[0]) + dy[0] + dp
                s = Mcsrch.max3(np.abs(theta), np.abs(dy[0]), np.abs(dp))
                gamma = s * np.sqrt(Mcsrch.sqr(theta / s) -
                                    (dy[0] / s) * (dp / s))
                if stp[0] > Mcsrch.sty[0]:
                    gamma = - gamma
                p = (gamma - dp) + theta
                q = ((gamma - dp) + gamma) + dy[0]
                r = p / q
                stpc = stp[0] + r * (Mcsrch.sty[0] - stp[0])
                stpf = stpc
            elif stp[0] > Mcsrch.stx[0]:
                stpf = stpmax
            else:
                stpf = stpmin

        # Update the interval of uncertainty. This update does not
        # depend on the new step or the case analysis above.

        if fp > Mcsrch.fx[0]:
            Mcsrch.sty[0] = stp[0]
            Mcsrch.fy[0] = fp
            dy[0] = dp
        else:
            if sgnd < 0.0:
                Mcsrch.sty[0] = Mcsrch.stx[0]
                Mcsrch.fy[0] = Mcsrch.fx[0]
                dy[0] = dx[0]
            Mcsrch.stx[0] = stp[0]
            Mcsrch.fx[0] = fp
            dx[0] = dp

        # Compute the new step and safeguard it.

        stpf = np.min([stpmax, stpf])
        stpf = np.max([stpmin, stpf])
        stp[0] = stpf

        if Mcsrch.brackt[0] and bound:
            if Mcsrch.sty[0] > Mcsrch.stx[0]:
                stp[0] = np.min(
                    [Mcsrch.stx[0] + 0.66 * (Mcsrch.sty[0] - Mcsrch.stx[0]), stp[0]])
            else:
                stp[0] = np.max(
                    [Mcsrch.stx[0] + 0.66 * (Mcsrch.sty[0] - Mcsrch.stx[0]), stp[0]])

        return
