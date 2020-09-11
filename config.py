class Config:
    def __init__(self, encoder, train):
        self.max_epochs = 64
        self.early_stop_time = 5
        self.min_check_epoch = 0
        self.vocab_num = 21128  # token number of bert-base-chinese
        self.max_len = 256
        self.encoder = encoder
        self.train = train
        self.attention_dim = 64
        self.lr = 1e-5 if encoder == 'bert' else 1e-3
        self.batch_size = 3 if encoder == 'bert' else 32
        self.word_embedding_dim = 128
        self.para_embedding_dim = 768
        self.filter_sizes = (3, 5, 7, 9)
        assert all([s%2 for s in self.filter_sizes])
        assert self.para_embedding_dim % len(self.filter_sizes) == 0
        self.num_filters = self.para_embedding_dim // len(self.filter_sizes)
    
    def store_name(self, task):
        return '{}_{}'.format(self.encoder, task)
    
    def parameter_info(self, task):
        obj = {'encoder': self.encoder, 'task': task}
        return obj