""" GAN configuration placeholder """


class GANConfiguration:
    """ GAN configuration placeholder """

    def __init__(self):
        self._name = None
        self._batch_size = 2048
        self._X_nodes = 29
        self._y_nodes = 1
        self._z_dims = 100
        self._X_name = 'input_X'
        self._z_name = 'z_prior'
        self._drop_out = False

    @property
    def name(self):
        """ name of the GAN"""
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def batch_size(self):
        """ Batch size """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def X_nodes(self):
        """ Number of X nodes """
        return self._X_nodes

    @X_nodes.setter
    def X_nodes(self, value):
        self._X_nodes = value

    @property
    def X_name(self):
        """ Name of tensorflow input X placeholder  """
        return self._X_name

    @X_name.setter
    def X_name(self, value):
        self._X_name = value

    @property
    def y_nodes(self):
        """ Number of y nodes """
        return self._y_nodes

    @y_nodes.setter
    def y_nodes(self, value):
        self._y_nodes = value

    @property
    def z_dims(self):
        """ Dimension of z noise vector """
        return self._z_dims

    @z_dims.setter
    def z_dims(self, value):
        self._z_dims = value

    @property
    def z_name(self):
        """  Name of noise tensorflow placeholder """
        return self._z_name

    @z_name.setter
    def z_name(self, value):
        self._z_name = value

    @property
    def drop_out(self):
        """  Drop out or not flag """
        return self._drop_out

    @drop_out.setter
    def drop_out(self, value):
        self._drop_out = value
