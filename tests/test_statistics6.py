import unittest
import numpy as np
from statistics6 import *

class StatisticsTests(unittest.TestCase):
    def test_hand_logloss(self):
        np.testing.assert_allclose(losses([1,0],[[.8,.2]]),-np.log(.8))
    def test_clipping(self):
        self.assertTrue(np.isfinite(losses([0,1],[[1,0]])).all())
    def test_not_probability_ensemble(self):
        value=losses([1],[[.1],[.9]]).mean()
        self.assertGreater(value,float(losses([1],[[.5]]).mean()))
    def test_pairing_requires_exact_ids_and_seeds(self):
        with self.assertRaises(ValueError):paired([1,0],[[.8,.2]],[[.8,.2]],[1,2],[2,1],[1],[1])
        with self.assertRaises(ValueError):paired([1,0],[[.8,.2]],[[.8,.2]],[1,2],[1,2],[1],[2])
    def test_seed_interval_is_not_match_pseudoreplication(self):
        a=paired([1,0],np.array([[.7,.3],[.8,.2],[.9,.1]]),np.full((3,2),.5),[1,2],[1,2],[1,2,3],[1,2,3])
        b=paired([1,0]*5,np.tile(np.array([[.7,.3],[.8,.2],[.9,.1]]),(1,5)),np.full((3,10),.5),list(range(10)),list(range(10)),[1,2,3],[1,2,3])
        self.assertAlmostEqual(a['L'],b['L']);self.assertEqual(a['df'],2)
    def test_all_decisions_and_strict_boundaries(self):
        self.assertEqual(decision(-.008,-.006,.005),'G_practical_gain')
        self.assertEqual(decision(.006,.008,.005),'N_practical_gain')
        self.assertEqual(decision(-.004,.004,.005),'operational_equivalence')
        self.assertEqual(decision(-.005,.004,.005),'unresolved')
        self.assertEqual(decision(-.006,.006,.005),'unresolved')
    def test_one_fit_has_no_seed_interval(self):
        self.assertIsNone(seed_interval([.2])['L'])
    def test_invalid_probabilities(self):
        for p in ([[np.nan]],[[1.2]],[[-.1]]):
            with self.assertRaises(ValueError):losses([1],p)
    def test_block_threshold_and_constant(self):
        a=block_sensitivity(np.ones(19),list(range(19)),replicates=50)
        self.assertIsNone(a['L'])
        b=block_sensitivity(np.ones(20),list(range(20)),replicates=50)
        self.assertEqual(b['L'],[1.]);self.assertEqual(b['U'],[1.])
    def test_unequal_cluster_sizes_use_match_weighting(self):
        a=block_sensitivity([0,0,0,1],['a','a','a','b'],replicates=100,minimum_groups=2)
        self.assertEqual(a['point'],[.25])
    def test_bootstrap_matches_explicit_cluster_draws(self):
        counts=np.arange(1,21);group_values=np.linspace(-.02,.03,20)
        x=np.repeat(group_values,counts);groups=np.repeat(np.arange(20),counts)
        result=block_sensitivity(x,groups,replicates=100,seed=731)
        weights=np.random.default_rng(731).multinomial(20,np.full(20,.05),size=100)
        manual=np.array([sum(w[j]*counts[j]*group_values[j] for j in range(20))/sum(w*counts) for w in weights])
        np.testing.assert_allclose(result['L'],[np.quantile(manual,.025)])
        np.testing.assert_allclose(result['U'],[np.quantile(manual,.975)])
    def test_bootstrap_pairing_same_draws(self):
        values=np.arange(30,dtype=float)
        a=block_sensitivity(np.stack([values,values+2],1),list(range(30)),replicates=100)
        self.assertAlmostEqual(a['L'][1]-a['L'][0],2)
        self.assertAlmostEqual(a['U'][1]-a['U'][0],2)
    def test_time_anchor_and_short_final_block(self):
        self.assertEqual(time_blocks([1772668800,1772668800+7*86400-1,1772668800+7*86400],7),[0,0,1])
    def test_equal_month(self):
        np.testing.assert_allclose(equal_group_seed_values([[0,0,1]],['a','a','b']),[.5])

if __name__=='__main__':unittest.main()
