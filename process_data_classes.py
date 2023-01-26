import numpy as np
from gym import spaces


class ProcessDataBase:
    def __init__(self, **kwargs):
        self.dim = kwargs.get('dim', 20)
        self.min_tbs = kwargs.get('min_tbs', 32)
        self.max_tbs = kwargs.get('max_tbs', 286976)
        self.min_snr = kwargs.get('min_snr', 3)
        self.max_snr = kwargs.get('max_snr', 30)
        self.min_mcs = kwargs.get('min_mcs', 0)
        self.max_mcs = kwargs.get('max_mcs', 27)
        self.max_power = kwargs.get('max_power', 50)
        self.norm_multiplier = kwargs.get('norm_multiplier', 1)
        tti = kwargs.get('tti', 1e-3)
        self.decision_period_time = kwargs.get('decision_period_time', 0.1)
        
        if 'max_n_users' not in kwargs:
            raise Exception('Input variable n_users missing.')
        tb_per_tti = kwargs.get('max_n_users')
        n_ttis = self.decision_period_time / tti + 1
        self.max_tb_per_period = tb_per_tti * n_ttis
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.dim,), dtype=np.float32)
        self.max_val = -1
        self.n_context_av = kwargs.get('n_context_av', 1)
        self.n_max_bss = kwargs.get('n_max_bss', 60)
        
    def process(self, stats, tb_data, queue_data):
        raise NotImplementedError
        
    def get_obs_space(self):
        return self.observation_space


class ProcessDataTbsPdf(ProcessDataBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.delta = kwargs.get('delta')

    def process(self, stats, tb_data, queue_data):
        tbss = tb_data['tbs']
        
        assert len(tbss) <= self.max_tb_per_period, 'The TBS list contains more element than the maximum number of TBs sent in the decision period'
        
        hist, _ = np.histogram(tbss, bins=self.dim, range=(self.min_tbs, self.max_tbs), density=False)
        norm_hist = hist / self.max_tb_per_period  * self.norm_multiplier
        self.max_val = np.maximum(self.max_val, np.max(norm_hist))
        
        obs = norm_hist
        
        norm_power = stats['total_energy_uJ']*1e-6/self.decision_period_time / self.max_power
        reward = norm_power + self.delta * (1 - stats['prob_dec_tb'])
        info = {}
        return obs, reward, info

class ProcessData3DPdf(ProcessDataBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.delta = kwargs.get('delta')

    def process(self, stats, tb_data, queue_data):
        tbss = tb_data['tbs']
        mcss = tb_data['mcs']
        snrs = tb_data['snr']
        
        assert len(tbss) == len(mcss) and len(tbss) == len(snrs)
        assert len(tbss) <= self.max_tb_per_period, 'The TB data contains more samples than the maximum number of TBs sent in the decision period'
        
        range_tbs = (self.min_tbs, self.max_tbs)
        range_mcs = (self.min_mcs, self.max_mcs)
        range_snr = (self.min_snr, self.max_snr)
        hist, _ = np.histogramdd((tbss, mcss, snrs),bins=(self.dim,self.dim,self.dim),range=(range_tbs,range_mcs,range_snr))
        
        norm_hist = hist / self.max_tb_per_period * self.norm_multiplier
        self.max_val = np.maximum(self.max_val, np.max(norm_hist))
        obs = norm_hist
        
        norm_power = stats['total_energy_uJ']*1e-6/self.decision_period_time / self.max_power
        reward = norm_power + self.delta * (1 - stats['prob_dec_tb'])
        info = {}
        return obs, reward, info


   
    
class ProcessDataTbsPdfMF(ProcessDataBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.delta = kwargs.get('delta')
        self.action_mf_dim = kwargs.get('action_mf_dim')
        self.context_list = [[] for _ in range(self.n_max_bss)]

    def process(self, stats, tb_data, queue_data):
        tbss = np.array(tb_data['tbs'])
        bs_idx = np.array(tb_data['bs_idx'])
        
        assert len(tbss) <= self.max_tb_per_period, 'The TBS list contains more element than the maximum number of TBs sent in the decision period'
        
        bs_idx_unique = np.unique(bs_idx)
        obs = []
        
        for bs in bs_idx_unique:
            tbss_bs_i = tbss[bs_idx == bs]

            hist_i, _ = np.histogram(tbss_bs_i, bins=self.dim, range=(self.min_tbs, self.max_tbs), density=False)
            norm_hist_i = hist_i / self.max_tb_per_period * self.norm_multiplier
            
            # average with previous contexts
            self.context_list[bs].append(norm_hist_i)
            if len(self.context_list[bs]) > self.n_context_av:
                del self.context_list[bs][0]
                
            context_i = np.zeros(norm_hist_i.shape)
            for prev_c in self.context_list[bs]:
                context_i += prev_c
            context_i /= len(self.context_list[bs])
                        
            obs.append(context_i)
            
            self.max_val = np.maximum(self.max_val, np.max(context_i))

        norm_power = stats['total_energy_uJ']*1e-6/self.decision_period_time / self.max_power
        reward = norm_power + self.delta * (1 - stats['prob_dec_tb'])
        info = {'active_bss' : bs_idx_unique}
        return obs, reward, info
    
    
    
class ProcessData3DPdfMF(ProcessDataBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.delta = kwargs.get('delta')
        self.action_mf_dim = kwargs.get('action_mf_dim')
        self.context_list = [[] for _ in range(self.n_max_bss)]

    def process(self, stats, tb_data, queue_data):
        tbss = np.array(tb_data['tbs'])
        mcss = np.array(tb_data['mcs'])
        snrs = np.array(tb_data['snr'])
        bs_idx = np.array(tb_data['bs_idx'])
        
        assert len(tbss) == len(mcss) and len(tbss) == len(snrs)
        assert len(tbss) <= self.max_tb_per_period, 'The TBS list contains more element than the maximum number of TBs sent in the decision period'
        range_tbs = (self.min_tbs, self.max_tbs)
        range_mcs = (self.min_mcs, self.max_mcs)
        range_snr = (self.min_snr, self.max_snr)
            
        bs_idx_unique = np.unique(bs_idx)
        obs = []
        
        for bs in bs_idx_unique:
            
            tbss_bs_i = tbss[bs_idx == bs]
            mcss_bs_i = mcss[bs_idx == bs]
            snrs_bs_i = snrs[bs_idx == bs]
            hist_i, _ = np.histogramdd((tbss_bs_i, mcss_bs_i, snrs_bs_i),bins=(self.dim,self.dim,self.dim),range=(range_tbs,range_mcs,range_snr))

            norm_hist_i = hist_i / self.max_tb_per_period * self.norm_multiplier
            
            # average with previous contexts
            self.context_list[bs].append(norm_hist_i)
            if len(self.context_list[bs]) > self.n_context_av:
                del self.context_list[bs][0]
                
            context_i = np.zeros(norm_hist_i.shape)
            for prev_c in self.context_list[bs]:
                context_i += prev_c
            context_i /= len(self.context_list[bs])
                        
            obs.append(context_i)
            
            self.max_val = np.maximum(self.max_val, np.max(context_i))

        norm_power = stats['total_energy_uJ']*1e-6/self.decision_period_time / self.max_power
        reward = norm_power + self.delta * (1 - stats['prob_dec_tb'])
        info = {'active_bss' : bs_idx_unique}
        return obs, reward, info

