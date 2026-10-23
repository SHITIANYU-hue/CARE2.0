import unittest, itertools, math, random, socket
from statistics import mean
from normalization import *

class NormalizationTests(unittest.TestCase):
    def test_exhaustive_max(self):
        for x in [(-5.,-2.,0.,4.),(1.,1.,1.),(-7.,-7.,-3.,-3.,8.),(0.,.001,100.),(2.,)]:
            for k in range(1,len(x)+1):
                exact=mean(max(s) for s in itertools.combinations(x,k))
                self.assertAlmostEqual(expected_max_gap(x,k),max(x)-exact,places=12)
    def test_full_process_enumeration(self):
        x=(-4.,-1.,-1.,2.,9.)
        for m,T in [(1,1),(1,3),(2,2),(3,2)]:
            exact=mean(trace_auc(path,m,T) for path in itertools.permutations(x,m+T))
            self.assertAlmostEqual(reference(x,m,T)['random_auc'],exact,places=12)
    def test_conditional_enumeration(self):
        x=(-4.,-1.,-1.,2.,9.)
        for ids in itertools.combinations(range(len(x)),2):
            rest=[x[i] for i in range(len(x)) if i not in ids];prefix=tuple(x[i] for i in ids)
            exact=mean(trace_auc(prefix+path,2,2) for path in itertools.permutations(rest,2))
            self.assertAlmostEqual(reference(x,2,2,ids)['random_auc'],exact,places=12)
    def test_affine(self):
        x=(-9.,-3.,1.,6.,12.);r=reference(x,2,2);s=reference([7*v+39 for v in x],2,2)
        self.assertAlmostEqual(score(4.,r),score(67.,s),places=12)
        self.assertAlmostEqual(paired_delta(4.,2.,r,r),paired_delta(67.,53.,s,s))
    def test_random_and_oracle(self):
        r=reference([-3.,0.,2.,8.,10.],2,2)
        self.assertAlmostEqual(score(r['random_auc'],r),0.)
        self.assertEqual(score(r['oracle'],r),1.)
        self.assertLess(score(-3.,r),0.)
    def test_saturation(self):
        for r in [reference([7.]*8,2,3),reference([0,1,1,1,1],1,3),reference([0,1,2,3],3,1),reference([0,1,2,3],1,2,[3])]:
            self.assertTrue(r['zero_headroom']);self.assertIsNone(score(r['oracle'],r))
    def test_near_zero_retained(self):
        r=reference([0]*5+[1]*15,2,2)
        self.assertTrue(r['near_saturation']);self.assertGreater(r['headroom'],0)
        self.assertIsNotNone(score(.9,r));self.assertLess(score(.9,r),0)
    def test_offset_cancellation(self):
        x=[1e12+i*.0001220703125 for i in range(20)]
        r=reference(x,5,10);self.assertGreater(r['headroom'],0)
    def test_invalid_inputs(self):
        for f in [lambda:reference([],1,1),lambda:reference([1,float('nan'),3],1,1),lambda:reference([1,2],2,1),lambda:reference([1,2,3],1,1,[3]),lambda:reference([1,2,3,4],2,1,[0,0]),lambda:reference([1,2,3],True,1),lambda:trace_auc([1,2],1,2)]:
            with self.assertRaises(ValueError):f()
    def test_mismatch_rejected(self):
        r=bind_reference(reference([0,1,2,3],1,2),'a'*64);s=bind_reference(reference([0,1,2,3],1,2),'b'*64)
        with self.assertRaises(ValueError):paired_delta(2,1,r,s)
    def test_auc_timing(self):
        self.assertEqual(trace_auc([5,1,3,7],2,2),6.)
    def test_sign(self):
        r=reference([-4,0,1,3,9],1,3)
        for a,b in [(1,2),(2,1),(2,2)]:self.assertEqual(math.copysign(1,paired_delta(a,b,r,r)),math.copysign(1,a-b))

if __name__=='__main__':unittest.main(verbosity=2)
