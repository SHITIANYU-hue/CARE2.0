import unittest
from . import pipeline_controller as c
class ControllerTests(unittest.TestCase):
 def test_stop_each_phase(self):
  for stop in range(5):
   called=[];states=[]
   def run(phase,command):
    called.append(phase);return 7 if len(called)==stop+1 else 0
   self.assertEqual(c.execute(run,states.append),7)
   self.assertEqual(called,list(c.PHASES[:stop+1]));self.assertEqual(states[-1]['state'],'stopped')
 def test_complete_order_and_local_commands(self):
  called=[];states=[]
  self.assertEqual(c.execute(lambda phase,cmd:called.append(phase) or 0,states.append),0)
  self.assertEqual(called,list(c.PHASES));self.assertEqual(states[-1]['state'],'complete')
  for phase,cmd in c.commands():
   self.assertEqual(cmd.count('generate'),int(phase=='generate'))
   if phase!='generate':self.assertIn('--output',cmd);self.assertEqual(cmd[-1],'4')
