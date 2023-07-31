import ConfigSpace as CS

from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, CategoricalHyperparameter
from ConfigSpace import ConfigurationSpace


def get_configspace(seed=42, with_hp=True):
        cs = ConfigurationSpace(seed=seed)
        n_conf_layer = UniformIntegerHyperparameter("n_conv_layers", 1, 3, default_value=3)
        n_conf_layer_0 = UniformIntegerHyperparameter("n_channels_conv_0", 32, 128, default_value=32)
        n_conf_layer_1 = UniformIntegerHyperparameter("n_channels_conv_1", 64, 256, default_value=64)
        n_conf_layer_2 = UniformIntegerHyperparameter("n_channels_conv_2", 64, 256, default_value=64)
        k_conf_layer_0 = UniformIntegerHyperparameter("kernel_conv_0", 0, 3, default_value=2)
        k_conf_layer_1 = UniformIntegerHyperparameter("kernel_conv_1", 0, 3, default_value=2)
        k_conf_layer_2 = UniformIntegerHyperparameter("kernel_conv_2", 0, 3, default_value=2)
        
        use_BN = CategoricalHyperparameter("use_BN", [False, True])
        global_avg_pooling = CategoricalHyperparameter("global_avg_pooling", [False, True])
        
        n_fc_layers = UniformIntegerHyperparameter("n_fc_layers", 0, 2)
        dropout_rate = UniformFloatHyperparameter("dropout_rate", 0.0, 0.5, default_value=0.2)


        if with_hp==True:
                learning_rate_init = UniformFloatHyperparameter('learning_rate_init',0.00001, 1.0, default_value=2.244958736283895e-05, log=True)
                standardize = CategoricalHyperparameter("standardize", [False, True])
                augment = CategoricalHyperparameter("augment", [False, True])
                cs.add_hyperparameters([learning_rate_init, augment,  standardize])
                                                        
        
        
        cs.add_hyperparameters([n_conf_layer, n_conf_layer_0, n_conf_layer_1, n_conf_layer_2,
                                k_conf_layer_0, k_conf_layer_1, k_conf_layer_2,
                                use_BN, global_avg_pooling, n_fc_layers, dropout_rate])

        use_conf_layer_2 = CS.conditions.InCondition(n_conf_layer_2, n_conf_layer, [3])
        use_conf_layer_1 = CS.conditions.InCondition(n_conf_layer_1, n_conf_layer, [2, 3])
        use_kernel_conf_layer_2 = CS.conditions.InCondition(k_conf_layer_2, n_conf_layer, [3])
        use_kernel_conf_layer_1 = CS.conditions.InCondition(k_conf_layer_1, n_conf_layer, [2, 3])
        
        cs.add_conditions([use_conf_layer_2, use_conf_layer_1, use_kernel_conf_layer_2, use_kernel_conf_layer_1])

        dimensions = len(cs.get_hyperparameters())

        return cs , dimensions


