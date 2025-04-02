#! /usr/bin/env python3
import sys
from TD.kernels import *
import numpy as np
import bisect
import unittest
from scipy.stats import norm, uniform
from scipy.integrate import quad
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad
from sklearn.neighbors import KernelDensity


"""
Annotations used for the autograder.

[START-AUTOGRADER-ANNOTATION]
{
  "total" : 6,
  "names" : [
      "kernels.py::test_radial", 
      "kernels.py::test_flat", 
      "kernels.py::test_gaussian",
      "kernels.py::test_knn", 
      "kernels.py::test_meanshift", 
      "kernels.py::test_last_section"
        ],
  "points" : [10, 10, 16, 16, 16, 16]
}
[END-AUTOGRADER-ANNOTATION]
"""


class Grader(unittest.TestCase):

    class Radtest(Radial):
        def volume(self: Self) -> float:
            return len(self.data)
        def profile(self: Self, t: float) -> float:
            return t
            
    def test_radial(self):
        mycloud=[np.array([1.0,2.0]),np.array([3.0,4.0]),np.array([0.0,-1.0]),np.array([3.0,7.0]),np.array([10.0,-2.0])]

        r=self.Radtest(2, mycloud, 0.42);

        p=np.array([0.,0.])
        self.assertAlmostEqual(r.density(p), 248.09621505, msg="radial-density")
        
        p=np.array([42.,-1.])
        self.assertAlmostEqual(r.density(p), 9782.4466143222, msg="radial-density")

        p=np.array([-3.,1.])
        self.assertAlmostEqual(r.density(p), 417.77860048025, msg="radial-density")

        p=np.array([3.,3.])
        self.assertAlmostEqual(r.density(p), 155.54218664034, msg="radial-density")

        self.assertTrue(issubclass(Radial,Kernel), msg="Radial should be inherited from Kernel")

    def test_flat(self):
        ker=Flat(7,[],5.00)
        self.assertAlmostEqual(ker.volume(),4.72476597033)
        self.assertAlmostEqual(ker.profile(0.5),1.0)
        self.assertAlmostEqual(ker.profile(1.5),0.0)
        self.assertAlmostEqual(ker.profile(2.5),0.0)
        self.assertAlmostEqual(ker.profile(0.0),1.0)
        self.assertAlmostEqual(ker.profile(0.999),1.0)
        self.assertTrue(issubclass(Flat,Radial), msg="Flat should be inherited from Radial")

    def test_gaussian(self):
        mycloud=[np.array([1.,2.,3.,4.,5.,6.,7.]),np.array([0.,-2.,3.,11.,-5.,6.,3.])]
        ker=Gaussian(7,mycloud,5.00)
        self.assertAlmostEqual(ker.volume(),621.7696785429)
        self.assertAlmostEqual(ker.profile(0.5),0.778800783071)
        self.assertAlmostEqual(ker.profile(1.5),0.472366552741)
        self.assertAlmostEqual(ker.profile(2.5),0.286504796860)
        self.assertAlmostEqual(ker.profile(5.0),0.08208499862)
 
        p=np.array([1.,1.,1.,1.,1.,1.,1.])
        self.assertAlmostEqual(ker.density(p)/1.9546911479e-9,1., places=4)

        p=np.array([0.,0.,0.,0.,0.,0.,0.])
        self.assertAlmostEqual(ker.density(p)/7.99962e-10,1., places=4)
        p3=np.array([2.,4.,6.,8.,10.,12.,14.])
        ker.guess_bandwidth()
        self.assertAlmostEqual(ker.bandwidth/8.320276142,1., places=4)
        self.assertTrue(issubclass(Gaussian,Radial), msg="Gaussian should be inherited from Radial")

    def test_knn(self):
        mycloud=[np.array([0.,0.]),np.array([1.,1.])]
        myknn=Knn(2, mycloud,3,1.)
        p=np.array([0.,1.])
        print(myknn.k_dist_knn(p,1))


        p=np.array([0.,0.5])
        self.assertAlmostEqual(myknn.k_dist_knn(p,1),0.5, places=4)
        p=np.array([0.5,0.5])
        self.assertAlmostEqual(myknn.k_dist_knn(p,1),0.70710678, places=4)
        p=np.array([2.,2.])
        self.assertAlmostEqual(myknn.k_dist_knn(p,1),1.414213562, places=4)
        p=np.array([4.,1.])
        myknn.data.append(p.copy())
        p=np.array([2.,3.])
        myknn.data.append(p.copy())
        myknn.fit_knn()
        self.assertAlmostEqual(myknn.density(p),0.1325825214724776, places=4)
        p=np.array([2.,2.])
        self.assertAlmostEqual(myknn.k_dist_knn(p,2),1.414213562, places=4)
        self.assertAlmostEqual(myknn.density(p),0.16770509831248423, places=4)
        p=np.array([2.,0.])
        self.assertAlmostEqual(myknn.k_dist_knn(p,2),2., places=4)
        p=np.array([2.,0.])
        self.assertAlmostEqual(myknn.k_dist_knn(p,3),2.2360679774, places=4)
        self.assertAlmostEqual(myknn.density(p),0.16770509831248423, places=4)

        self.assertTrue(issubclass(Knn,Kernel), msg="Knn should be inherited from Kernel")

    def test_meanshift(self):
        myrawdata = np.loadtxt("csv/normal.csv", delimiter=" ", dtype=float)
        myknn=Knn(2,myrawdata,3, 1.)
        myknn.meanshift(3)
        self.assertAlmostEqual(myknn.data[0][0], -0.742035, places=4)
        self.assertAlmostEqual(myknn.data[0][1], 1.09605, places=4)
        self.assertAlmostEqual(myknn.data[1][0], 0.025417, places=4)
        self.assertAlmostEqual(myknn.data[1][1],  0.1229, places=4)
        self.assertAlmostEqual(myknn.data[2][0], 0.845654, places=4)
        self.assertAlmostEqual(myknn.data[2][1], -0.97052, places=4)
        self.assertAlmostEqual(myknn.data[3][0], 0.04139, places=4)
        self.assertAlmostEqual(myknn.data[3][1], 0.921662, places=4)
        self.assertAlmostEqual(myknn.data[4][0], 0.00173367, places=4)
        self.assertAlmostEqual(myknn.data[4][1], -0.848124, places=4)
        self.assertAlmostEqual(myknn.data[100][0], 0.663376, places=4)
        self.assertAlmostEqual(myknn.data[100][1], 0.107646, places=4)


    def test_last_section(self):
        def generate_data_2d(n_samples:int, index:int, density_type="gaussian_mixture"):
            if density_type == "gaussian_mixture":
                if index==1:
                    weights = [0.6, 0.4]
                    means = [[-2, 2], [2, -2]]
                    covs = [[[1, 0.], [0., 1]], [[1, -0.], [-0., 1]]]  # Matrices de covariance
                if index==2:
                    weights = [0.6, 0.4]
                    means = [[-2, 2], [2, -2]]
                    covs = [[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]]]  # Matrices de covariance
                if index==3:
                    weights = [0.3, 0.3,0.4]
                    means = [[-2, 2], [2, -2], [3, 0]]
                    covs = [[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]], [[18, -0.5], [-0.5, 60]]]  # Matrices de covariance
                if index==4:
                    weights = [0.3, 0.2,0.2,0.3]
                    means = [[-2, 2], [2, -2], [3, 0], [0, 0]]
                    covs = [[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]], [[1, -0.5], [-0.5, 5]], [[1, -3], [-3, 10]]]  # Matrices de covariance
                data = np.concatenate([
                    np.random.multivariate_normal(mean=means[i], cov=covs[i], size=int(n_samples * weights[i]))
                    for i in range(len(weights))
                ])
                true_density = lambda x, y: np.sum([weights[i] * multivariate_normal.pdf([x, y], mean=means[i], cov=covs[i]) for i in range(len(weights))])

            elif density_type == "uniform":
                data = np.random.uniform(low=-3, high=3, size=(n_samples, 2))
                true_density = lambda x, y: uniform.pdf(x, loc=-3, scale=6) * uniform.pdf(y, loc=-3, scale=6)

            else:  # Default to 2D standard normal
                data = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=n_samples)
                true_density = lambda x, y: multivariate_normal.pdf([x, y], mean=[0, 0], cov=[[1, 0], [0, 1]])

            return data, true_density


        def total_variation_distance_2d(p, q, lower=-np.inf, upper=np.inf):
            integrand = lambda x, y: np.abs(q(x, y) - p(x, y))
            tv_distance, _ =   dblquad(integrand, lower, upper, lambda x: lower, lambda x: upper, epsabs=0.5)  # scipy.integrate.dblquad
            return 0.5 *tv_distance

        intervalles=[0.9,1.2,1.8,4]
        notes=["A+", "A", "B", "C", "D"]

        grades=[4,4,4,4]
        for job in [1,2,3,4]:

            n_samples = [100,100,200,250][job-1]

            data, true_density = generate_data_2d(n_samples, index=job)
            kde = KernelDensity(bandwidth='silverman', kernel='gaussian')
            kde.fit(data)
            estimated_density_sk = lambda x, y: np.exp(kde.score_samples(np.array([[x, y]])))[0]
            tv_distance_sk = total_variation_distance_2d(true_density, estimated_density_sk)
            kg = Gaussian(2,data, 1.)
            estimated_density = lambda x, y: kg.density(np.array([x, y]))

            kg.guess_bandwidth_challenge()

            estimated_density = lambda x, y: kg.density(np.array([x, y]))
            tv_distance = total_variation_distance_2d(true_density, estimated_density)


            ratio=tv_distance / tv_distance_sk
            note=bisect.bisect_left(intervalles,ratio)
            grades[job-1]=note

        grades.sort()
        if grades[3]==0:  # if worst note is A+ then final note is A+
            final=0
        if grades[3]==1: # if worst note is A then final note is A
            final=1
        else:
            final=3  # else final note is second worst note
        print(f"Grade:{notes[final]}")

def print_help():
    print(
        "./grader script. Usage: ./grader.py test_number, e.g., ./grader.py 1 for the 1st exercise."
    )
    print("N.B.: ./grader.py 0 runs all tests.")
    print(f"You provided {sys.argv}.")
    exit(1)



def suite(test_nb):
    suite = unittest.TestSuite()
    test_name = [
        "test_radial",
        "test_flat",
        "test_gaussian",
        "test_knn", 
        "test_meanshift",
        "test_last_section",
    ]

    if test_nb > 0:
        suite.addTest(Grader(test_name[test_nb - 1]))
    else:
        for name in test_name:
            suite.addTest(Grader(name))

    return suite

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
    try:
        test_nb = int(sys.argv[1])
    except ValueError as e:
        print(
            f"You probably didn't pass an int to ./grader.py: passed {sys.argv[1]}; error {e}"
        )
        exit(1)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite(test_nb))
