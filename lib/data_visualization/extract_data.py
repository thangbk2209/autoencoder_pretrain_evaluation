from pandas import read_csv
import pandas as pd
import numpy as np


class GridExtractor:
	def __init__(self, time_interval=None, file_data_path=None):
		self.time_interval = time_interval
		self.file_data_path = file_data_path
		self.colnames = ['job_id', 'submit_time', 'wait_time', 'run_time', 'n_proc', 'used_cpu_time', 'used_memory',
						 'req_n_procs', 'req_time', 'req_memory', 'status', 'user_id', 'group_id', 'excutabled',
						 'queue_id', 'partition_id', 'orig_site_id', 'last_run_site_id', 'job_structure',
						 'job_structure_params', 'used_networks', 'used_local_disk_space', 'used_resources',
						 'req_platform', 'req_network', 'Req_local_disk_space', 'req_resources', 'void', 'project_id']
		self.top_users_job = ['U66', 'U84', 'U239', 'U173', 'U6', 'U139', 'U69', 'U0', 'U193', 'U265']
		self.top_users_cpu_time = ['U86', 'U66', 'U19', 'U80', 'U267', 'U9', 'U173', 'U223', 'U118', 'U8']
		self.df = read_csv(self.file_data_path, header=None, index_col=False, names=self.colnames, engine='python')
		self.new_colnames = ['job_id_data', 'submit_time_data', 'wait_time_data', 'run_time_data', 'start_point',
							 'end_point', 'n_proc_data', 'used_cpu_time_data', 'used_memory_data', 'user_id_data',
							 'group_id_data']

	def compute_start_finish_point(self):
		self.job_id_data = self.df['job_id'].values
		self.submit_time_data = self.df['submit_time'].values
		self.wait_time_data = self.df['wait_time'].values
		self.run_time_data = self.df['run_time'].values
		self.n_proc_data = self.df['n_proc'].values
		self.used_cpu_time_data = self.df['used_cpu_time'].values
		self.used_memory_data = self.df['used_memory'].values
		self.user_id_data = self.df['user_id'].values
		self.group_id_data = self.df['group_id'].values
		# print(job_id_data.shape, submit_time_data.shape)
		self.start_point = self.submit_time_data + self.wait_time_data
		self.end_point = self.start_point + self.run_time_data
		data = np.array([self.job_id_data, self.submit_time_data, self.wait_time_data, self.run_time_data,
						 self.start_point, self.end_point, self.n_proc_data, self.used_cpu_time_data,
						 self.used_memory_data, self.user_id_data, self.group_id_data])
		data_df = pd.DataFrame(np.transpose(data))
		data_df.to_csv(self.file_data_path.rsplit('/', 1)[0] + '/useful_anons_job.csv', index=False, header=None)
		self.new_file_data_path = self.file_data_path.rsplit('/', 1)[0] + '/useful_anons_job.csv'

	def extract_information(self):
		print(self.job_id_data.shape)
		print(self.job_id_data[0])
		job_id_series = []
		n_process_data_series = []
		used_cpu_time_series = []
		used_memory_series = []
		user_id_series = []
		group_id_series = []
		start_batch_monitor = self.start_point[0]
		while True:
			end_batch_monitor = start_batch_monitor + self.time_interval
			_job_id = []
			_n_process_data = 0
			_used_cpu_time = 0
			_used_memory = 0
			_user_id = []
			_group_id = []
			
			for i in range(len(self.job_id_data)):
				if self.start_point[i] >= start_batch_monitor and self.start_point[i] < end_batch_monitor:
					if self.job_id_data[i] not in _job_id:
						_job_id.append(self.job_id_data[i])
					_n_process_data += 1
					_used_cpu_time += self.used_cpu_time_data[i]
					_used_memory += self.used_memory_data[i]
					if self.user_id_data[i] not in _user_id:
						_user_id.append(self.user_id_data[i])
					if self.group_id_data[i]:
						_group_id.append(self.group_id_data[i])
				elif self.start_point[i] >= end_batch_monitor:
					print(self.start_point[i])
					print('break for loop')
					print(_job_id, _used_cpu_time)
					print('end_batch_monitor', start_batch_monitor, end_batch_monitor)
					job_id_series.append(len(_job_id))
					n_process_data_series.append(_n_process_data)
					used_cpu_time_series.append(_used_cpu_time)
					used_memory_series.append(_used_cpu_time)
					user_id_series.append(len(_user_id))
					group_id_series.append(len(_group_id))
					break
			start_batch_monitor = end_batch_monitor
			print('breaked for loop')
			print(start_batch_monitor, self.end_point[-1])
			if start_batch_monitor > self.end_point[-1]:
				print('break while loop')
				break
		data = np.array([job_id_series, n_process_data_series, used_cpu_time_series, used_memory_series,
						 user_id_series, group_id_series])
		data_df = pd.DataFrame(np.transpose(data))
		data_df.to_csv(self.file_data_path.rsplit('/', 1)[0] + '/timeseries_anonjobs_{}Min.csv'\
			.format(self.time_interval / 60), index=False, header=None)