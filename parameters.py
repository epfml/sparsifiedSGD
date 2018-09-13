class Parameters:
    """
    Parameters class used for all the experiments, redefine a string representation to summarize the experiment
    """

    def __init__(self,
                 num_epoch,
                 lr_type,
                 initial_lr=None,
                 regularizer=None,
                 epoch_decay_lr=None,
                 take_k=None,
                 take_top=False,
                 with_memory=False,
                 estimate='final',
                 name=None,
                 n_cores=1,
                 tau=None,
                 real_update_every=1,
                 qsgd_s=None):
        # a lot of sanity checks to fail fast if we have inconsistent parameters
        assert num_epoch >= 0
        assert lr_type in ['constant', 'epoch-decay', 'decay', 'bottou']

        if lr_type in ['constant', 'decay']:
            assert initial_lr > 0
        if lr_type == 'decay':
            assert initial_lr and tau
            assert regularizer > 0
        if lr_type == 'epoch-decay':
            assert epoch_decay_lr is not None

        if not take_k:
            assert not take_top and not with_memory

        assert estimate in ['final', 'mean', 't+tau', '(t+tau)^2']

        if qsgd_s is not None:
            assert take_k is None and real_update_every == 1

        assert n_cores >= 1

        self.num_epoch = num_epoch
        self.lr_type = lr_type
        self.initial_lr = initial_lr
        self.regularizer = regularizer
        self.epoch_decay_lr = epoch_decay_lr
        self.take_k = take_k
        self.take_top = take_top
        self.with_memory = with_memory
        self.estimate = estimate
        self.name = name
        self.n_cores = n_cores
        self.tau = tau
        self.real_update_every = real_update_every
        self.qsgd_s = qsgd_s

    def __str__(self):
        if self.name:
            return self.name

        lr_str = self.lr_str()
        sparse_str = self.sparse_str()

        reg_str = ""
        if self.regularizer:
            reg_str = "-reg{}".format(self.regularizer)

        return "epoch{}-{}{}-{}-{}".format(self.num_epoch, lr_str, reg_str, sparse_str, self.estimate)

    def lr_str(self):
        lr_str = ""
        if self.lr_type == 'constant':
            lr_str = "lr{}".format(self.initial_lr)
        elif self.lr_type == 'decay':
            lr_str = "lr{}decay{}".format(self.initial_lr, self.epoch_decay_lr)
        elif self.lr_type == 'custom':
            lr_str = "lr{}/lambda*(t+{})".format(self.initial_lr, self.tau)
        elif self.lr_type == 'bottou':
            lr_str = "lr-bottou-{}".format(self.initial_lr)
        else:
            lr_str = "lr-{}".format(self.lr_type)

        return lr_str

    def sparse_str(self):
        if not self.take_k:
            sparse_str = "full"
        else:
            if self.take_top:
                sparse_str = "top{}".format(self.take_k)
            else:
                sparse_str = "rand{}".format(self.take_k)

            if self.with_memory:
                sparse_str += "-with-mem"
            else:
                sparse_str += "-no-mem"
        return sparse_str

    def __repr__(self):
        return "Parameter('{}')".format(str(self))
