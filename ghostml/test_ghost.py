#
# Copyright (C) 2021 Greg Landrum and GHOST contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import unittest

import numpy as np
import pandas as pd
import pickle
import ghost
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

class TestGHOST(unittest.TestCase):
    def test_regression1(self):
        thresholds = np.round(np.arange(0.05,0.55,0.05),2)
        random_seed = 16

        with open('test_data/chembl3371_testing_data.pkl','rb') as inf:
            tpl = pickle.load(inf)

        fps_train, fps_test, labels_train, labels_test, names_train, names_test = tpl    
        # train classifier
        cls = GradientBoostingClassifier(n_estimators = 100, validation_fraction = 0.2, n_iter_no_change = 10, 
                                        tol = 0.01, random_state=random_seed)
        cls.fit(fps_train, labels_train)

        # predict the train set
        train_probs = cls.predict_proba(fps_train)[:,1] #prediction probabilities for the train set
        # predict the test set
        test_probs = cls.predict_proba(fps_test)[:,1] #prediction probabilities for the test set
        #store predictions in dataframe
        scores = [1 if x>=0.5 else 0 for x in test_probs]
        df_preds = pd.DataFrame({'mol_names': names_test, 'y_true': labels_test, 'standard': scores})

        # generate and show some evaluation stats for the model on the test data:

        auc = metrics.roc_auc_score(labels_test, test_probs)
        kappa = metrics.cohen_kappa_score(labels_test,scores)
        self.assertAlmostEqual(kappa,0.731,places=3)
        self.assertAlmostEqual(auc,0.943,places=3)

        # optimize the decision thresholds based on the prediction probabilities of N training subsets
        # Can be used for every machine learning model
        thresh_sub = ghost.optimize_threshold_from_predictions( labels_train, train_probs, thresholds,
                                                                ThOpt_metrics = 'Kappa', 
                                                                N_subsets = 100, subsets_size = 0.2, 
                                                                with_replacement = False, random_seed = random_seed) 
        self.assertAlmostEqual(thresh_sub,0.25,places=2)
        scores = [1 if x>=thresh_sub else 0 for x in test_probs]
        kappa = metrics.cohen_kappa_score(labels_test,scores)
        self.assertAlmostEqual(kappa,0.784,places=3)

if __name__=='__main__':
    unittest.main()
