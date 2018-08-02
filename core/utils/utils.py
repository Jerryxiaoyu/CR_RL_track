import numpy as np
import torch
import os
from torch.autograd import Variable
import shutil
import glob
import csv
from datetime import datetime
import pandas as pd


def _grad_norm(net):
	norm = 0.0
	for param in net.parameters():
		if param.grad is not None:
			norm += np.linalg.norm(param.grad.data.cpu().numpy())
	
	return norm


def save_net_param(net, save_dir , name ='network', mode ='param'):
	# save only the parameters
	if mode =='param':
		torch.save(net.state_dict(), os.path.join(save_dir, name+'.pkl'))
	elif mode == 'net':
		torch.save(net , os.path.join(save_dir, name+'.pkl'))
	else:
		assert print('Model Save mode is error! ')


def configure_log_dir(logname, txt='', copy = False, log_group=None, No_time = False):
	"""
	Set output directory to d, or to /tmp/somerandomnumber if d is None
	"""
	if log_group is not None:
		root_path = os.path.join('log-files', log_group)
	else:
		root_path = os.path.join('log-files' )
	
	if No_time:
		path = os.path.join(root_path, logname,   txt)
	else:
		now = datetime.now().strftime("%b-%d_%H:%M:%S")
		path = os.path.join(root_path, logname, now+txt)
	os.makedirs(path)  # create path
	if copy:
		filenames = glob.glob('*.py')  # put copy of all python files in log_dir
		for filename in filenames:  # for reference
			shutil.copy(filename, path)
	return path



class Logger(object):
	""" Simple training logger: saves to file and optionally prints to stdout """
	def __init__(self, logdir,csvname = 'log'):
		"""
		Args:
			logname: name for log (e.g. 'Hopper-v1')
			now: unique sub-directory name (e.g. date/time string)
		"""
		self.path = os.path.join(logdir, csvname+'.csv')
		self.write_header = True
		self.log_entry = {}
		self.f = open(self.path, 'w')
		self.writer = None  # DictWriter created with first call to write() method

	def write(self, display=True):
		""" Write 1 log entry to file, and optionally to stdout
		Log fields preceded by '_' will not be printed to stdout

		Args:
			display: boolean, print to stdout
		"""
		if display:
			self.disp(self.log_entry)
		if self.write_header:
			fieldnames = [x for x in self.log_entry.keys()]
			self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
			self.writer.writeheader()
			self.write_header = False
		self.writer.writerow(self.log_entry)
		self.log_enbtry = {}

	@staticmethod
	def disp(log):
		"""Print metrics to stdout"""
		log_keys = [k for k in log.keys()]
		log_keys.sort()
		'''
		print('***** Episode {}, Mean R = {:.1f} *****'.format(log['_Episode'],
															   log['_MeanReward']))
		for key in log_keys:
			if key[0] != '_':  # don't display log items with leading '_'
				print('{:s}: {:.3g}'.format(key, log[key]))
		'''
		print('log writed!')
		print('\n')

	def log(self, items):
		""" Update fields in log (does not write to file, used to collect updates.

		Args:
			items: dictionary of items to update
		"""
		self.log_entry.update(items)

	def close(self):
		""" Close log file - log cannot be written after this """
		self.f.close()

	def log_table2csv(self,data, header = True):
		df = pd.DataFrame(data)
		df.to_csv(self.path, index=False, header=header)


	def log_csv2table(self):
		data = pd.read_csv(self.path,header = 0,encoding='utf-8')
		return np.array(data)


