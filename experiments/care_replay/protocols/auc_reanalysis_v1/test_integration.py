import unittest,json,gzip,math,hashlib
from pathlib import Path
from table_analysis import pool_binding
P=Path(__file__).resolve().parent;B=P.parents[1];OUT=B/'outputs/auc_reanalysis_v1'
class IntegrationTests(unittest.TestCase):
 def test_all_32_certificates_and_tables(self):
  cert=json.loads((OUT/'certification_report.json').read_text());tables=json.loads((OUT/'prepared_tables_report.json').read_text());self.assertEqual(len(cert['certified']),32);self.assertEqual(len(tables['tables']),32)
  mapping={r['id']:r for r in cert['certified']};n=0
  for row in tables['tables']:
   c=mapping[row['id']];path=B/c['pool_file'];data=path.read_bytes();self.assertEqual(hashlib.sha256(data).hexdigest(),c['pool_sha256']);pool=json.loads(data)
   self.assertEqual(pool_binding(pool['ordered_ids'],pool['values']),c['ordered_ID_value_sha256']);self.assertEqual(len(pool['ordered_ids']),c['n'])
   raw=(B/row['prepared_file']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),row['prepared_sha256']);rows=json.loads(gzip.decompress(raw));self.assertEqual(len(rows),row['logical_rows']);n+=len(rows)
   initials=json.loads((OUT/'initials'/f"{row['id']}.json").read_text());ids=set(pool['ordered_ids']);seen={}
   for r in rows:
    self.assertEqual(r['pool_binding'],c['ordered_ID_value_sha256']);self.assertTrue(math.isfinite(r['auc']));self.assertEqual((r['initial'],r['rounds']),(c['initial'],c['rounds']))
    start=initials[str(r['seed'])];self.assertEqual(len(set(start)),r['initial']);self.assertTrue(set(start)<=ids)
    key=(r['seed'],r['physical_mode'])
    if key in seen:self.assertEqual(r['auc'],seen[key])
    seen[key]=r['auc']
  self.assertEqual(n,254160)
 def test_no_real_normalization_output(self):
  self.assertFalse((OUT/'normalized_results').exists());self.assertFalse(json.loads((OUT/'certification_report.json').read_text())['real_normalized_scores_computed'])
 def test_raw_permissions_restored(self):
  p=B/'outputs/experiment3_recruitment_v1/sealed/DMS_ProteinGym_substitutions.zip';self.assertEqual(p.stat().st_mode&0o777,0)
 def test_config_approved_constraints(self):
  c=json.loads((P/'config.json').read_text());self.assertEqual(c['model_api_calls_allowed'],0);self.assertFalse(c['clip']);self.assertEqual(c['near_saturation_fraction'],.01);self.assertEqual(c['bootstrap_seed'],202610218101)
if __name__=='__main__':unittest.main(verbosity=2)
