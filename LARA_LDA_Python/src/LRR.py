import numpy as np

from LRR_Model import *
from Vector4Review import *


class LRR:
    def __init__(self, aspect_model, corpus, alpha_step, alpha_tol, beta_step, beta_tol, lamb):
        self.aspect_model = aspect_model
        self.corpus = corpus

        self.PI = 0.5
        self.K = 7

        self.m_alpha_step = alpha_step
        self.m_alpha_tol = alpha_tol
        self.m_beta_step = beta_step
        self.m_beta_tol = beta_tol
        self.m_lambda = lamb
        self.m_v = corpus.V
        self.m_k = aspect_model.m_k
        self.m_train_size = 0
        self.m_test_size = 0
        self.m_collection = []

        self.m_diag_beta = np.zeros(self.m_k * (self.m_v+1), dtype=np.float64)  # to include the bias term for each aspect
        self.m_g_beta = np.zeros(len(self.m_diag_beta), dtype=np.float64)
        self.m_beta = np.zeros(len(self.m_g_beta), dtype=np.float64)

        self.m_diag_alpha = np.zeros(self.m_k, dtype=np.float64)
        self.m_g_alpha = np.zeros(self.m_k, dtype=np.float64)
        self.m_alpha = np.zeros(self.m_k, dtype=np.float64)
        self.m_alpha_cache = np.zeros(self.m_k, dtype=np.float64)

        self.init_collection_from_corpus()
        self.m_model = LRR_Model(self.m_k, self.m_v)
        self.m_old_alpha = np.zeros(self.m_k)  # in case optimization for alpha failed

    def init_collection_from_corpus(self):
        for rest in self.corpus.Restaurants:
            for review in rest.Reviews:
                is_train = np.random.rand() < 0.75
                if is_train:
                    self.m_train_size += 1
                else:
                    self.m_test_size += 1
                ratings = np.random.rand(self.m_k)*5
                ratings = np.insert(ratings, 0, review.Overall)
                v4review = Vector4Review(review, ratings, is_train, self.aspect_model)
                v4review.normalize()
                self.m_collection.append(v4review)

    def init(self, voca_size):
        if len(self.m_collection) == 0:
            print("[Error]Load training data first!")
            return -1

    def em_estimation(self, max_iter, converge):
        iter = 0
        diff = 10
        likelihood = 0
        old_likelihood = self.MStep(False)

        while iter < np.min(8, max_iter) or (iter < max_iter and diff > converge):
            alpha_exp = 0
            alpha_cov = 0

            # E-step
            for vct in self.m_collection:
                if vct.m_4train:
                    tag = self.EStep(vct)
                    if tag == -1:  # failed to converge
                        alpha_cov += 1
                    elif tag == -2:  # failed with exceptions
                        alpha_exp += 1
            print("{}\t".format(iter))

            # M-step
            likelihood = self.MStep(iter % 4 == 3)

            self.evaluate_aspect()  # evaluating in the testing cases
            diff = (old_likelihood-likelihood)/old_likelihood
            old_likelihood = likelihood

            iter += 1

    def EStep(self, vct):
        # step 1: estimate aspect rating
        vct.get_aspect_rating(self.m_model.m_beta)

        # step 2: infer aspect weight
        try:
            self.m_old_alpha = vct.m_alpha.copy()
            return self.infer_alpha(vct)
        except Exception as e:
            self.m_alpha = self.m_old_alpha.copy()
            return -2

    def infer_alpha(self, vct):
        f = 0
        iprint = [-1, 0]
        iflag = [0]
        icall = 0
        n = self.m_model.m_k
        m = 5

        # initialize the diagonal matrix
        self.m_diag_alpha.fill(0)

        while True:
            f = self.get_alpha_obj_gradient(vct)
            # lbfgs(n, m, vct.m_alpha_hat, f, self.m_g_alpha, False, self.m_diag_alpha, iprint, self.m_alpha_tol, 1e-20, iflag)

            icall += 1
            if not iflag[0] != 0 and icall <= self.m_alpha_step:
                break

    def get_alpha_obj_gradient(self, vct):
        expsum = Utilities.exp_sum(vct.m_alpha_hat)
        overall_rating = -vct.m_ratings[0]
        _sum = 0

        # initialize the gradient
        self.m_g_alpha.fill(0)

        for i in range(self.m_k):
            vct.m_alpha = np.exp(vct.m_alpha_hat[i])/expsum  # map to aspect weight

            overall_rating += vct.m_alpha[i] * vct.m_aspect_rating[i]  # estimate the overall rating
            self.m_alpha_cache[i] = vct.m_alpha_hat[i] - self.m_model.m_mu[i]  # difference with prior

            s = self.PI * np.power(vct.m_aspect_rating[i]-vct.m_ratings[0], 2)

            if np.abs(s) > 1e-10:  # in case we will disable it
                for j in range(self.m_model.m_k):
                    if j == i:
                        self.m_g_alpha[j] += 0.5 * s * vct.m_alpha[i] * (1-vct.m_alpha[i])
                    else:
                        self.m_g_alpha[j] -= 0.5 * s * vct.m_alpha[i] * vct.m_alpha[i]

                _sum += vct.m_alpha[i] * s

        diff = overall_rating / self.m_model.m_delta
        for i in range(self.m_k):
            s = 0
            for j in range(self.m_k):
                # part I of objective function: data likelihood
                if i == j:
                    self.m_g_alpha[j] += diff*vct.m_aspect_rating[i] * vct.m_alpha[i] * (1-vct.m_alpha[i])
                else:
                    self.m_g_alpha[j] -= diff * vct.m_aspect_rating[i] * vct.m_alpha[i] * vct.m_alpha[i]

                # part II of objective function: prior
                s += self.m_alpha_cache[j] * self.m_model.m_sigma_inv[i][j]

            self.m_g_alpha[i] += s
            _sum += self.m_alpha_cache[i] * s

        return 0.5 * (overall_rating * overall_rating / self.m_model.m_delta + sum)

    # m-step can only be applied to training samples!!
    def MStep(self, update_sigma=False):
        k = self.m_k

        # Step 0: initialize the statistics
        self.m_g_alpha.fill(0)

        # Step 1: ML for \mu
        for vct in self.m_collection:
            if vct.m_4train == False:
                continue

            for i in range(self.m_k):
                self.m_g_alpha[i] += vct.m_alpha_hat[i]

        for i in range(self.m_k):
            self.m_model.m_mu[i] = self.m_g_alpha[i] / self.m_train_size
        self.test_alpha_variance(update_sigma)

        # Step 2: ML for \sigma
        # update_sigma = False

        # calculate the likelihood for the alpha part
        alpha_likelihood = 0
        beta_likelihood = 0
        for vct in self.m_collection:
            if vct.m_4train == False:
                continue

            for i in range(k):
                self.m_diag_alpha[i] = vct.m_alpha_hat[i] - self.m_model.m_mu[i]
            alpha_likelihood += self.m_model.calc_covariance(self.m_diag_alpha)
        alpha_likelihood += np.log(self.m_model.calc_det())

        # Step 3: ML for \beta
        self.ml_beta()

        beta_likelihood = self.get_beta_prior_obj()

        # Step 4: ML for \delta
        datalikelihood = self.get_data_likelihood()
        auxdata = self.get_aux_data_likelihood()
        old_delta = self.m_model.m_delta
        self.m_model.m_delta = datalikelihood / self.m_train_size
        datalikelihood /= old_delta

        return alpha_likelihood + beta_likelihood + datalikelihood + auxdata + np.log(self.m_model.m_delta)

    def test_alpha_variance(self, update_sigma):
        # test the variance of \hat\alpha estimation
        self.m_diag_alpha.fill(0)

        v = 0
        for vct in self.m_collection:
            if vct.m_4train == False:
                continue

            for i in range(self.m_k):
                v = vct.m_alpha_hat[i] - self.m_model.m_mu[i]
                self.m_diag_alpha[i] += v * v  # just for variance

        for i in range(self.m_k):
            self.m_diag_alpha[i] /= self.m_train_size
            if i == 0 and update_sigma:
                print("*")

            # mean and variance of \hat\alpha
            print("{}:{}\t".format(self.m_model.m_mu[i], self.m_diag_alpha[i]))

    def ml_beta(self):
        f = 0
        iprint = [-1, 0]
        iflag = [0]
        icall = 0
        n = (1+self.m_model.m_v) * self.m_model.m_k
        m = 5

        for i in range(self.m_k):
            pos = i*(self.m_model.m_v+1)
            for j in range(self.m_model.m_v+1):
                self.m_beta[j+pos] = self.m_model.m_beta[i][j]

        self.m_diag_beta.fill(0)
        while True:
            if icall%1000 == 0:
                print(".")  # keep track of beta update
            f = self.get_beta_obj_gradient()  # to be minimized
            # lbfgs(n, m, self.m_beta, f, self.m_g_beta, False, self.m_diag_beta, iprint, self.m_beta_tol, 1e-20, iflag)

            icall += 1
            if not (iflag[0] != 0 and icall <= self.m_beta_step):
                break

        print(icall + "\t")
        for i in range(self.m_model.m_k):
            pos = i * (self.m_model.m_v + 1)
            for j in range(self.m_model.m_v + 1):
                self.m_model.m_beta[i] = self.m_beta[j + pos]

        return f

    # \beat^T * \beta
    def get_beta_prior_obj(self):
        likelihood = 0
        for i in range(len(self.m_model.m_beta)):
            for j in range(len(self.m_model.m_beat[i])):
                likelihood += self.m_model.m_beta[i][j] * self.m_model.m_beta[i][j]

        return self.m_lambda * likelihood

    # \sum_d(\sum_i\alpha_{di}\S_{di}-r_d)^2/\sigma^2
    def get_data_likelihood(self):
        likelihood = 0

        # part I of objective function: data likelihood
        for vct in self.m_collection:
            if vct.m_4train == False:
                continue  # do not touch testing cases

            orating = -vct.m_ratings[0]

            # apply the current model
            vct.get_aspect_rating(self.m_model.m_beta)
            for i in range(len(vct.m_alpha)):
                orating += vct.m_alpha[i] * vct.m_aspect_rating[i]
            likelihood += orating * orating

        return likelihood

    # \sum_d\pi\sum_i\alpha_{di}(\S_{di}-r_d)^2
    def get_aux_data_likelihood(self):
        likelihood = 0

        # part I of objective function: data likelihood
        for vct in self.m_collection:
            if vct.m_4train == False:
                continue  # do not touch testing cases

            orating = vct.m_ratings[0]
            for i in range(len(vct.m_alpha_hat)):
                likelihood += vct.m_alpha[i] * (vct.m_aspect_rating[i] - orating) * (vct.m_aspect_rating[i] - orating)

        return self.PI * likelihood

    def evaluate_aspect(self):
        aMSE = 0
        oMSE = 0
        icorr = 0
        acorr = 0
        i = -1
        iError = False
        aError = False

        pred = np.zeros((self.m_k, self.m_test_size), dtype=np.float64)
        ans = np.zeros((self.m_k, self.m_test_size), dtype=np.float64)

        for vct in self.m_collection:
            if vct.m_4train:
                continue  # only evaluating in testing cases
            i += 1

            diff = self.prediction(vct) - vct.m_ratings[0]
            oMSE += diff * diff
            for j in range(self.m_k):
                pred[j][i] = vct.m_aspect_rating[j]
                ans[j][i] = vct.m_ratings[j+1]

            # 1. Aspect evaluation: to skip overall rating in ground-truth
            aMSE += Utilities.MSE(vct.m_aspect_rating, vct.m_ratings, 1)
            corr = Utilities.correlation(vct.m_aspect_rating, vct.m_ratings, 1)

            if np.isnan(corr) == False:
                icorr += corr
            else:
                print("Error occur")
                iError = True  # error occur

        # 2. entity level evaluation
        for j in range(self.m_k):
            corr = Utilities.correlation(pred[j], ans[j], 0)
            if np.isnan(corr) == False:
                acorr += corr
            else:
                aError = True

        # MSE for overall rating, MSE for aspect rating, item level correlation, aspect level correlation
        if iError:
            print('x')
        else:
            print('o')
        if aError:
            print('x')
        else:
            print('o')
        print("{}\t{}\t{}\t{}".format(
            np.sqrt(oMSE/self.m_test_size),
            np.sqrt(aMSE/self.m_test_size),
            (icorr/self.m_test_size),
            (acorr/self.m_k)))

    def prediction(self, vct):
        # predict aspect rating
        vct.get_aspect_rating_with_v(self.m_beta, self.m_v+1)
        orating = 0
        for i in range(self.m_k):
            orating += self.m_alpha_cache[i] * vct.m_aspect_rating[i]
        return orating

    def get_beta_obj_gradient(self):
        likelihood = 0
        aux_likelihood = 0
        vSize = self.m_model.m_v + 1

        # initialize the structure
        self.m_g_beta.fill(0)

        # part I of objective function: data likelihood
        for vct in self.m_collection:
            if vct.m_4train == False:
                continue  # do not touch testing cases

            oRate = vct.m_ratings[0]
            orating = -oRate

            # apply the current model
            vct.get_aspect_rating_with_v(self.m_beta, vSize)
            for i in range(self.m_model.m_k):
                orating += vct.m_alpha[i] * vct.m_aspect_rating[i]

            likelihood += orating*orating
            orating /= self.m_model.m_delta

            offset = 0
            for i in range(self.m_model.m_k):
                aux_likelihood += vct.m_alpha[i] * (vct.m_aspect_rating[i]-oRate) * (vct.m_aspect_rating[i]-oRate)
                diff = vct.m_alpha[i] * (orating * self.PI*(vct.m_aspect_rating[i] - oRate)) * vct.m_aspect_rating[i]

            sVct = vct.m_aspectV[i]
            self.m_g_beta[offset] += diff
            for j in range(len(sVct.m_index)):
                self.m_g_beta[offset + sVct.m_index[j]] += diff * sVct.m_value[j]
            offset += vSize  # move to next aspect

        reg = 0
        for i in range(len(self.m_beta)):
            self.m_g_beta[i] += self.m_lambda * self.m_beta[i]
            reg += self.m_beta[i] * self.m_beta[i]

        return 0.5 * (likelihood/self.m_model.m_delta + self.PI*aux_likelihood + self.m_lambda*reg)

