import numpy as np

from src.ExceptionWithIflag import *
from src.LBFGS_M import LBFGS
from src.LRR_Model import *
from src.Vector4Review import *


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

        # to include the bias term for each aspect
        self.m_diag_beta = np.zeros(self.m_k * (self.m_v+1), dtype=np.float64)
        self.m_g_beta = np.zeros(len(self.m_diag_beta), dtype=np.float64)
        self.m_beta = np.zeros(len(self.m_g_beta), dtype=np.float64)

        self.m_diag_alpha = np.zeros(self.m_k, dtype=np.float64)
        self.m_g_alpha = np.zeros(self.m_k, dtype=np.float64)
        self.m_alpha = np.zeros(self.m_k, dtype=np.float64)
        self.m_alpha_cache = np.zeros(self.m_k, dtype=np.float64)

        self.init_collection_from_corpus()
        self.m_model = LRR_Model(self.m_k, self.m_v)
        # in case optimization for alpha failed
        self.m_old_alpha = np.zeros(self.m_k)

    def init_collection_from_corpus(self):
        for rest in self.corpus.Restaurants:
            for review in rest.Reviews:
                is_train = np.random.rand() < 0.75
                if is_train:
                    self.m_train_size += 1
                else:
                    self.m_test_size += 1
                # ratings = np.ones(self.m_k)*int(review.Overall)
                ratings = np.random.rand(self.m_k)*5
                ratings = np.insert(ratings, 0, review.Overall)
                # ratings = np.zeros(1+self.m_k)
                # ratings[0] = int(review.Overall)
                # for i in range(self.m_k):
                #     # sample from N(overall_score, sigma=2)
                #     score = int(review.Overall) + 2 + np.random.randn()
                #     ratings[1+i] = np.min([np.max([0, score]), 5])
                v4review = Vector4Review(
                    review, ratings, is_train, self.aspect_model)
                v4review.normalize()
                # print("[NORMALIZE] {} {}".format(v4review.m_ID, v4review.m_alpha))
                self.m_collection.append(v4review)

    def init_with_v(self, v):
        if v == -1:
            return

        if self.m_collection is None or len(self.m_collection) == 0:
            print("[Error]Load training data first!")
            return -1

        vct = self.m_collection[0]
        self.m_v = v
        self.m_k = len(vct.m_aspectV)

        self.m_diag_beta = np.zeros(self.m_k * (self.m_v+1), dtype=np.float64)
        self.m_g_beta = np.zeros(len(self.m_diag_beta), dtype=np.float64)
        self.m_beta = np.zeros(len(self.m_g_beta), dtype=np.float64)

        self.m_diag_alpha = np.zeros(self.m_k, dtype=np.float64)
        self.m_g_alpha = np.zeros(self.m_k, dtype=np.float64)
        self.m_alpha = np.zeros(self.m_k, dtype=np.float64)
        self.m_alpha_cache = np.zeros(self.m_k, dtype=np.float64)

        return 0

    def init(self, v):
        self.init_with_v(v)
        initV = 1
        if len(self.m_collection) == 0:
            print("[Error]Load training data first!")
            return -1

        return initV

    def load_from_file(self, modelfile):
        self.m_collection.clear()
        self.m_model = LRR_Model(-1, -1)
        self.m_model.load_from_file(modelfile)
        self.m_old_alpha = np.zeros(self.m_model.m_k)

    def load_vectors(self, vector_file):
        with open(vector_file, 'r') as f:
            self.m_train_size = 0
            self.m_test_size = 0

            is_train = False
            count = 0
            len = 0

            while True:
                tmpTxt = f.readline()
                if not tmpTxt:
                    break

                # is_train == (count % 4 < 3) # Warning
                mod = count % 4
                is_train = (mod < 3)
                if is_train:
                    self.m_train_size += 1
                else:
                    self.m_test_size += 1
                count += 1
                arr = tmpTxt.split('\t')
                vct = Vector4Review(None, None, None, None)
                vct.init_from_file(arr[0], arr[1:], is_train)

                for i in range(self.m_k):
                    tmpTxt = f.readline().strip()
                    arr = tmpTxt.split(" ")
                    vct.add_aspect(arr)
                vct.normalize()

                self.m_collection.append(vct)
                len = np.max([vct.get_length(), len])  # max index word

        print("Load {} restaurants data", count)
        return len

    def em_estimation(self, max_iter, converge, v):
        iter = 0
        diff = 10
        old_likelihood = self.init(v)

        while iter < max_iter or (iter < max_iter and diff > converge):
            alpha_exp = 0
            alpha_cov = 0

            # E-step
            count = 1
            for vct in self.m_collection:
                if vct.m_4train:
                    tag = self.EStep(vct)
                    if tag == -1:  # failed to converge
                        alpha_cov += 1
                    elif tag == -2:  # failed with exceptions
                        alpha_exp += 1
                    count += 1
            print("[Iteration] {}".format(iter))

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
        self.m_old_alpha = vct.m_alpha.copy()
        ret = self.infer_alpha(vct)
        if ret == -2:
            vct.m_alpha = self.m_old_alpha.copy()
            return -2

        return ret

    def infer_alpha(self, vct):
        f = 0
        iprint = [-1, 0]
        iflag = [0]
        icall = 0
        n = self.m_model.m_k
        m = 5

        # initialize the diagonal matrix
        self.m_diag_alpha.fill(0)

        optimizer = LBFGS()
        while True:
            f = self.get_alpha_obj_gradient(vct)
            ret = optimizer.lbfgs_func(n, m, vct.m_alpha_hat, f, self.m_g_alpha,
                                       False, self.m_diag_alpha, iprint, self.m_alpha_tol, 1e-20, iflag)
            if ret == -1:
                return -2

            icall += 1
            if not (iflag[0] != 0 and icall <= self.m_alpha_step):
                break

        if iflag[0] != 0:
            return -1  # have not converged yet
        else:
            expsum = Utilities.exp_sum(vct.m_alpha_hat)
            for n in range(self.m_model.m_k):
                vct.m_alpha[n] = np.exp(vct.m_alpha_hat[n])/expsum
            return f

    def get_alpha_obj_gradient(self, vct):
        expsum = Utilities.exp_sum(vct.m_alpha_hat)
        overall_rating = -vct.m_ratings[0]
        _sum = 0

        # initialize the gradient
        self.m_g_alpha.fill(0)

        for i in range(self.m_k):
            vct.m_alpha[i] = np.exp(vct.m_alpha_hat[i]) / \
                expsum  # map to aspect weight

            # estimate the overall rating
            overall_rating += vct.m_alpha[i] * vct.m_aspect_rating[i]
            self.m_alpha_cache[i] = vct.m_alpha_hat[i] - \
                self.m_model.m_mu[i]  # difference with prior

            s = self.PI * \
                (vct.m_aspect_rating[i]-vct.m_ratings[0]) * \
                (vct.m_aspect_rating[i]-vct.m_ratings[0])

            if np.abs(s) > 1e-10:  # in case we will disable it
                for j in range(self.m_model.m_k):
                    if j == i:
                        self.m_g_alpha[j] += 0.5 * s * \
                            vct.m_alpha[i] * (1-vct.m_alpha[i])
                    else:
                        # Warning: not i
                        self.m_g_alpha[j] -= 0.5 * s * \
                            vct.m_alpha[i] * vct.m_alpha[j]

                _sum += vct.m_alpha[i] * s

        diff = overall_rating / self.m_model.m_delta
        for i in range(self.m_k):
            s = 0
            for j in range(self.m_k):
                # part I of objective function: data likelihood
                if i == j:
                    self.m_g_alpha[j] += diff*vct.m_aspect_rating[i] * \
                        vct.m_alpha[i] * (1-vct.m_alpha[i])
                else:
                    self.m_g_alpha[j] -= diff * vct.m_aspect_rating[i] * \
                        vct.m_alpha[i] * vct.m_alpha[j]

                # part II of objective function: prior
                s += self.m_alpha_cache[j] * self.m_model.m_sigma_inv[i][j]

            self.m_g_alpha[i] += s
            _sum += self.m_alpha_cache[i] * s

        return 0.5 * (overall_rating * overall_rating / self.m_model.m_delta + _sum)

    # m-step can only be applied to training samples!!
    def MStep(self, update_sigma=False):
        k = self.m_k

        # Step 0: initialize the statistics
        self.m_g_alpha.fill(0)

        # Step 1: ML for \mu
        count = 0
        for n, vct in enumerate(self.m_collection):
            if vct.m_4train == False:
                continue

            # print("hat {}: {}\t{}".format(count, self.m_g_alpha[0], vct.m_alpha_hat[0]))
            count += 1
            for i in range(self.m_k):
                self.m_g_alpha[i] += vct.m_alpha_hat[i]

        for i in range(self.m_k):
            self.m_model.m_mu[i] = self.m_g_alpha[i] / self.m_train_size
        self.test_alpha_variance(update_sigma)

        # Step 2: ML for \sigma
        # update_sigma = False

        # calculate the likelihood for the alpha part
        alpha_likelihood = 0
        count = 0
        for vct in self.m_collection:
            if vct.m_4train == False:
                continue

            for i in range(k):
                self.m_diag_alpha[i] = vct.m_alpha_hat[i] - \
                    self.m_model.m_mu[i]
            alpha_likelihood += self.m_model.calc_covariance(self.m_diag_alpha)
            # print("alpha {}: {}".format(count, alpha_likelihood))
            count += 1
        alpha_likelihood += np.log(self.m_model.calc_det())

        # Step 3: ML for \beta
        try:
            self.ml_beta()
        except ExceptionWithIflag as e:
            print(e)

        beta_likelihood = self.get_beta_prior_obj()

        # Step 4: ML for \delta
        datalikelihood = self.get_data_likelihood()
        auxdata = self.get_aux_data_likelihood()
        old_delta = self.m_model.m_delta
        self.m_model.m_delta = datalikelihood / self.m_train_size
        datalikelihood /= old_delta

        total = alpha_likelihood + beta_likelihood + \
            datalikelihood + auxdata + np.log(self.m_model.m_delta)
        print("[MStep] total={}, alpha_likelihood={}, beta_likelihood={}, data_likelihood={}, auxdata={}, delta={}".format(
            total, alpha_likelihood, beta_likelihood, datalikelihood, auxdata, np.log(self.m_model.m_delta)))
        return total

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
                print("[MStep] update_sigma")

            # mean and variance of \hat\alpha
            # print("[MStep] aspect={}, mu={}, diag_alpha={}\t".format(i, self.m_model.m_mu[i], self.m_diag_alpha[i]))

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
        optimizer = LBFGS()
        while True:
            # if icall%1000 == 0:
            #     print("[MStep] ml_beta update: icall={}".format(icall))  # keep track of beta update
            f = self.get_beta_obj_gradient()  # to be minimized
            ret = optimizer.lbfgs_func(n, m, self.m_beta, f, self.m_g_beta, False,
                                       self.m_diag_beta, iprint, self.m_beta_tol, 1e-20, iflag)
            if ret == -1:
                return

            icall += 1
            if not (iflag[0] != 0 and icall <= self.m_beta_step):
                break

        # print("[MStep] ml_beta update: icall={}".format(icall))
        for i in range(self.m_model.m_k):
            pos = i * (self.m_model.m_v + 1)
            for j in range(self.m_model.m_v + 1):
                self.m_model.m_beta[i] = self.m_beta[j + pos]

        return f

    # \beat^T * \beta
    def get_beta_prior_obj(self):
        likelihood = 0
        for i in range(len(self.m_model.m_beta)):
            for j in range(len(self.m_model.m_beta[i])):
                likelihood += self.m_model.m_beta[i][j] * \
                    self.m_model.m_beta[i][j]

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
                likelihood += vct.m_alpha[i] * (vct.m_aspect_rating[i] - orating) * (
                    vct.m_aspect_rating[i] - orating)

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
                if i == 0 and j == 0:
                    print("k={}, y={}, y_hat={}".format(
                        j, ans[j][i], pred[j][i]))

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
        # if iError:
        #     print('[Evaluation] iError=x (Error)')
        # else:
        #     print('[Evaluation] iError=o (No Error)')
        # if aError:
        #     print('[Evaluation] aError=x (Error)')
        # else:
        #     print('[Evaluation] aError=o (No Error)')
        # print("[Evaluation] oMSE={}, aMSE={}, icorr={}, acorr={}".format(
        #     np.sqrt(oMSE/self.m_test_size),
        #     np.sqrt(aMSE/self.m_test_size),
        #     (icorr/self.m_test_size),
        #     (acorr/self.m_k)))

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
                aux_likelihood += vct.m_alpha[i] * (
                    vct.m_aspect_rating[i]-oRate) * (vct.m_aspect_rating[i]-oRate)
                diff = vct.m_alpha[i] * (orating * self.PI*(
                    vct.m_aspect_rating[i] - oRate)) * vct.m_aspect_rating[i]

                sVct = vct.m_aspectV[i]
                self.m_g_beta[offset] += diff
                for j in range(len(sVct.m_index)):
                    self.m_g_beta[offset + sVct.m_index[j]
                                  ] += diff * sVct.m_value[j]
                offset += vSize  # move to next aspect

        reg = 0
        for i in range(len(self.m_beta)):
            self.m_g_beta[i] += self.m_lambda * self.m_beta[i]
            reg += self.m_beta[i] * self.m_beta[i]

        return 0.5 * (likelihood/self.m_model.m_delta + self.PI*aux_likelihood + self.m_lambda*reg)

    def print_prediction(self):
        for vct in self.m_collection:
            # all the ground-truth ratings
            y = ""
            for i in range(len(vct.m_ratings)):
                y += str(round(vct.m_ratings[i], 2)) + " "

            # predicted ratings
            vct.get_aspect_rating_with_v(self.m_beta, (1+self.m_v))
            y_hat = ""
            for i in range(len(vct.m_aspect_rating)):
                y_hat += str(round(vct.m_aspect_rating[i], 2)) + " "

            # inferred weights ( not meaningful for baseline logistic regression)
            aspect_weight = ""
            for i in range(len(vct.m_aspect_rating)):
                aspect_weight += str(round(vct.m_alpha[i], 2)) + " "

            print("original ratings={}, aspect_rating={}, aspect_weight={}".format(
                y, y_hat, aspect_weight))
