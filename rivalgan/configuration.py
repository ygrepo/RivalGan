""" Configuration place holder for the pipeline """


class Configuration:
    """ Configuration place holder for the pipeline """

    def __init__(self):
        self.info = {
            'APP': 'Fraud',
            'DATA_FOLDER': '../../data/',
            # 'DATA_FOLDER': '/home/ubuntu/insight/data/',
            'CACHE_FOLDER': '../../cache/',
            # 'CACHE_FOLDER': '/home/ubuntu/insight/cache/',
            'LOG_FOLDER': '../../logs/',
            # 'LOG_FOLDER': '/home/ubuntu/insight/logs/',
            'EMB_FOLDER': '../../emb_logs/',
            'MODEL_DIR': '../../models/',
            # 'MODEL_DIR': '/home/ubuntu/insight/models/',
            'DATA_FILE_NAME': 'creditcard.csv',
            'PICKLED_DATA': 'creditcard.engineered.pkl',
            # 'PICKLED_DATA': 'MaxAbsScaler_creditcard.engineered.pkl',
            # 'PICKLED_DATA': 'StandardScaler_creditcard.engineered.pkl',
            'MODEL_NAMES': ['Logit', 'LinearSVC', 'RandomForest', 'xgb'],
            'CORR_COLUMN_NAMES': ['V13', 'V15', 'V22', 'V23', 'V26', 'Amount'],
            'CLASS_NAME': 'Class',
            'CLASS_COLUMN_NAME': ['Class'],
            'FILTERED_COLUMN_NAMES': ['Class'],
            'AUGMENTED_DATA_SIZE': 5000,
            'TRAINING_INCR_STEP': 1,
            'TOTAL_TRAINING_STEPS': 100,
            'CLASSIFIER': 'SGDClassifier',
            # 'SAMPLER': 'SMOTETomek',
            'GAN_NAME': 'VGAN',
            'Z_DIM': 100,
            'Y_OUTPUT': 1,
            'SAMPLE': False,
            'NUM_TRAINING_STEPS': 10,
            'SEED': 42,
            'REAL': 0,
            'FAKE': 1,
            'PCA': True
        }

    def get(self, parameter_name):
        """ getter for a configuration parameter """
        if parameter_name in self.info:
            return self.info[parameter_name]
        return None

    def set(self, parameter_name, value):
        """ setter for a configuration parameter """
        self.info[parameter_name] = value

    def set_configuration(self, dargs):
        """initialize the configuration with a dict. of parameters  """
        for key, value in dargs.items():
            if isinstance(value, float):
                continue
            if value is not None:
                self.set(key, value)

    def __repr__(self):
        """ toString method """
        strl = []
        for key, value in self.info.items():
            strl.append(key + ':' + str(value) + "\n")
        return ''.join(strl)
