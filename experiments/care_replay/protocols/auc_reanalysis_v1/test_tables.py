import unittest,socket
from normalization import reference
from table_analysis import *

class TableTests(unittest.TestCase):
 def fixture(self):
  pool={'ordered_ids':['a','b','c','d'],'values':[-3.,0.,1.,4.]};pool['certified_binding']=pool_binding(pool['ordered_ids'],pool['values'])
  rows=[dict(seed=7,mode=m,auc=a,initial=1,rounds=2,pool_binding=pool['certified_binding']) for m,a in [('A',1.),('B',2.)]]
  return pool,rows,{'7':['a']}
 def test_shared_reference(self):
  p,r,i=self.fixture();out=transform_rows(p,r,i,1,2);a,b=out['rows']
  self.assertEqual(a['full_process_headroom'],b['full_process_headroom']);self.assertEqual(a['conditional_headroom'],b['conditional_headroom'])
  self.assertEqual(a['raw_auc'],1.)
 def test_reject_wrong_budget(self):
  p,r,i=self.fixture();r[0]['initial']=2
  with self.assertRaises(ValueError):transform_rows(p,r,i,1,2)
 def test_reject_changed_pool(self):
  p,r,i=self.fixture();p['values'][0]=99.
  with self.assertRaises(ValueError):transform_rows(p,r,i,1,2)
 def test_duplicate_rows(self):
  p,r,i=self.fixture()
  with self.assertRaises(ValueError):transform_rows(p,r+r,i,1,2)
 def test_preserve_interval(self):
  ref=reference([0,1,2,3,4],1,2);r=transform_existing_interval(1,-2,3,ref)
  self.assertAlmostEqual(r['low'],-2/ref['headroom']);self.assertLess(r['low'],0)
 def test_equal_groups_not_routes(self):
  rows=[dict(route='a',group='X',delta_score=0),dict(route='b',group='X',delta_score=0),dict(route='c',group='Y',delta_score=1)]
  a=equal_group_summary(rows,replicates=100);b=equal_group_summary(rows,replicates=100)
  self.assertEqual(a,b);self.assertEqual(a['mean'],.5)
 def test_no_silent_NA_drop(self):
  r=equal_group_summary([dict(route='a',group='X',delta_score=None),dict(route='b',group='Y',delta_score=1)],replicates=100)
  self.assertIsNone(r['mean']);self.assertEqual(r['routes'],2)
 def test_network_block(self):
  with self.assertRaises(RuntimeError):socket.getaddrinfo('example.invalid',443)
  with self.assertRaises(RuntimeError):socket.create_connection(('example.invalid',443))
if __name__=='__main__':unittest.main(verbosity=2)
