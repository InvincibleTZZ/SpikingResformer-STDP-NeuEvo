from collections import namedtuple

import torch

Genotype = namedtuple('Genotype', 'normal normal_concat')

"""
Operation sets
"""

PRIMITIVES = [
    'conv_3x3_p',
    # 'max_pool_3x3',
    # 'avg_pool_3x3',
    # 'def_conv_3x3',
    # 'def_conv_5x5',
    # 'sep_conv_3x3',
    # 'sep_conv_5x5',
    # 'dil_conv_3x3',
    # 'dil_conv_5x5',

    # 'max_pool_3x3_p',
    # 'avg_pool_3x3_p',
    'conv_3x3_p',
    'conv_5x5_p',
    # 'conv_3x3_p_p',
    # 'sep_conv_3x3_p',
    # 'sep_conv_5x5_p',
    # 'dil_conv_3x3_p',
    # 'dil_conv_5x5_p',
    # 'def_conv_3x3_p',
    # 'def_conv_5x5_p',n

    # 'max_pool_3x3_n',
    # 'avg_pool_3x3_n',
    'conv_3x3_n',
    'conv_5x5_n',
    # 'conv_3x3_p_n',
    # 'sep_conv_3x3_n',
    # 'sep_conv_5x5_n',
    # 'dil_conv_3x3_n',
    # 'dil_conv_5x5_n',
    # 'def_conv_3x3_n',
    # 'def_conv_5x5_n',

    # 新增的注意力操作
    'spike_self_attn_p',
    'spike_self_attn_n',
    'spike_cbam_p',
    'spike_cbam_n',
    'spike_channel_attn_p',
    'spike_channel_attn_n',
    'spike_spatial_attn_p',
    'spike_spatial_attn_n',

    # 新增的二次卷积操作
    'quad_conv_3x3',
    'quad_conv_5x5',
    'quad_conv_3x3_p',
    'quad_conv_5x5_p',
    'quad_conv_3x3_n',
    'quad_conv_5x5_n',
    
    # 二次卷积的back连接操作
    'quad_conv_3x3_back',
    'quad_conv_5x5_back',
    'quad_conv_3x3_p_back',
    'quad_conv_5x5_p_back',
    'quad_conv_3x3_n_back',
    'quad_conv_5x5_n_back',
    
    # 普通卷积的back连接操作
    'conv_3x3_n_back',
    'conv_5x5_n_back',
    
    # 注意力机制的back连接操作
    'spike_self_attn_n_back',
    'spike_cbam_n_back',
    'spike_cbam_p_back',

    # 'transformer',
]
"""====== SnnMlp Archirtecture By Other Methods"""

mlp1 = Genotype(
    normal=[
        ('mlp', 0), ('conv_3x3_p', 1),  # 2
        ('mlp', 1), ('mlp', 0),  # 3
        ('conv_3x3_p', 2), ('mlp', 3),  # 4
        #('mlp_back', 2),
        #('conv_3x3_p_back', 2)
    ],
    normal_concat=range(2, 5)
)

mlp2 = Genotype(
    normal=[
        ('mlp', 0), ('conv_3x3_p', 1),
        ('conv_3x3_p', 2), ('mlp_p', 1),
        # ('mlp_n', 1), ('conv_3x3_p', 2),
        ('mlp_back', 2)
    ],
    normal_concat=range(2, 4)
)


"""====== SNN Archirtecture By Other Methods"""

dvsc10_new_skip22 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_3x3_p', 0),  # 2
        ('conv_5x5_p', 1), ('conv_3x3_p', 2),  # 3
        ('conv_3x3_p', 0), ('conv_3x3_p', 3),  # 4
        ('conv_3x3_n_back', 2), ('conv_3x3_p_back', 3)  # 3, 4
    ],
    normal_concat=range(2, 5)
)

dvsc10_new_skip22 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_3x3_p', 0),
        ('conv_5x5_n', 1), ('conv_3x3_p', 2),
        ('conv_5x5_n', 0), ('conv_3x3_p', 3),
        ('conv_3x3_n_back', 0), ('conv_3x3_p_back', 1)
    ],
    normal_concat=range(2, 5)
)



dvsc10_new_skip20 = Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_5x5_n', 1),
        ('conv_3x3_n', 2), ('conv_5x5_p', 0),
        ('conv_3x3_p', 2), ('conv_3x3_n', 3),
        ('conv_3x3_p_back', 2),
        ('conv_5x5_p_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsc10_new_skip19 = Genotype(
    normal=[
        ('conv_5x5_n', 0), ('conv_3x3_p', 1),
        ('conv_5x5_n', 2), ('conv_5x5_n', 0),
        ('conv_3x3_p', 2), ('conv_5x5_p', 3),
        ('conv_3x3_p_back', 2),
        ('conv_5x5_p_back', 2)
    ],
    normal_concat=range(2, 5)
)

dvsc10_new_skip18 = Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_3x3_p', 1),
        ('conv_5x5_p', 2), ('conv_5x5_n', 0),
        ('conv_3x3_p', 2), ('conv_5x5_p', 3),
        ('conv_5x5_n_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip17 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_5x5_n', 0),
        ('conv_5x5_n', 2), ('conv_5x5_p', 1),
        ('conv_3x3_p', 2), ('avg_pool_3x3_p', 3),
        ('avg_pool_3x3_p_back', 2), ('conv_3x3_p_back', 2)
    ],
    normal_concat=range(2, 5)
)
dvsc10_new_skip16 = Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_5x5_n', 1),
        ('conv_3x3_n', 2), ('avg_pool_3x3_p', 0),
        ('conv_3x3_p', 2), ('avg_pool_3x3_n', 3),
        ('conv_3x3_p_back', 2),
        ('conv_3x3_p_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsc10_new_skip15 = Genotype(
    normal=[
        ('conv_5x5_n', 0), ('conv_5x5_p', 1),
        ('conv_5x5_p', 2), ('conv_5x5_n', 1),
        ('conv_3x3_p', 2), ('conv_5x5_p', 3),
        ('conv_5x5_p_back', 2),
        ('conv_3x3_p_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsc10_new_skip14 = Genotype(
    normal=[
        ('conv_5x5_n', 1), ('conv_3x3_p', 0),
        ('conv_5x5_p', 1), ('conv_3x3_p', 2),
        ('conv_3x3_p', 2), ('conv_5x5_n', 1),
        ('conv_3x3_n_back', 2),
        ('conv_3x3_p_back', 3)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip13 = Genotype(
    normal=[
        ('conv_5x5_n', 1), ('conv_3x3_p', 0),
        ('conv_5x5_n', 1), ('conv_3x3_p', 2),
        ('conv_3x3_p', 2), ('conv_5x5_n', 1),
        ('conv_3x3_n_back', 2),
        ('conv_3x3_p_back', 3)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip12 = Genotype(
    normal=[
        ('conv_5x5_n', 0), ('conv_3x3_p', 1),
        ('conv_5x5_n', 2), ('conv_5x5_n', 0),
        ('conv_3x3_p', 2), ('conv_5x5_p', 3),
        ('conv_3x3_n_back', 2),
        ('conv_3x3_p_back', 2)
    ],
    normal_concat=range(2, 5)
)
# dvsc10_new_skip12 = Genotype(
#     normal=[
#         ('conv_3x3_p', 0), ('conv_3x3_n', 1),
#         ('conv_3x3_p', 1), ('conv_5x5_n', 2),
#         ('conv_3x3_n', 3), ('conv_3x3_p', 0),
#         ('conv_5x5_p_back', 2), ('conv_3x3_p_back', 3)
#     ],
#     normal_concat=range(2, 5)
# )

dvsc10_new_skip11 = Genotype(normal=[
    ('conv_3x3_n', 0), ('conv_5x5_n', 1),
    ('conv_5x5_p', 0), ('conv_3x3_n', 2),
    ('conv_3x3_p', 2), ('conv_5x5_n', 0),
    ('conv_3x3_n_back', 2),
    ('conv_3x3_p_back', 3)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip10 = Genotype(
    normal=[
        ('conv_5x5_n', 1), ('conv_3x3_p', 0),
        ('conv_5x5_p', 2), ('conv_5x5_p', 1),
        ('conv_3x3_p', 2), ('conv_5x5_n', 1),
        ('conv_3x3_n_back', 2),
        ('conv_3x3_n_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip9 = Genotype(
    normal=[
        ('conv_5x5_p', 1), ('conv_5x5_n', 0),
        ('conv_5x5_p', 2), ('conv_5x5_n', 0),
        ('conv_3x3_p', 2), ('conv_5x5_p', 3),
        ('conv_3x3_p_back', 2),
        ('conv_5x5_n_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsc10_new_skip8 = Genotype(
    normal=[
        ('conv_5x5_n', 0), ('conv_5x5_n', 1),
        ('conv_3x3_n', 2), ('conv_5x5_p', 0),
        ('conv_3x3_p', 2), ('conv_5x5_n', 1),
        ('conv_5x5_n_back', 2),
        ('conv_3x3_p_back', 3)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip7 = Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_5x5_p', 1),
        ('conv_3x3_n', 2), ('conv_5x5_n', 0),
        ('conv_3x3_p', 2), ('conv_5x5_p', 3),
        ('conv_5x5_n_back', 2),
        ('conv_3x3_p_back', 3)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip6 = Genotype(
    normal=[
        ('conv_3x3_p', 0), ('conv_5x5_n', 1),
        ('conv_5x5_p', 2), ('conv_5x5_p', 1),
        ('conv_3x3_p', 2), ('conv_5x5_n', 0),
        ('conv_3x3_n_back', 2), ('conv_3x3_n_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip5 = Genotype(
    normal=[
        ('conv_3x3_p', 0), ('conv_3x3_p', 1),
        ('conv_3x3_n', 2), ('conv_3x3_n', 0),
        ('conv_3x3_p', 2), ('conv_3x3_p', 3),
        ('conv_5x5_n_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip4 = Genotype(
    normal=[
        ('conv_5x5_n', 1), ('conv_5x5_p', 0),
        ('conv_3x3_p', 2), ('conv_5x5_p', 1),
        ('conv_3x3_p', 2), ('conv_5x5_n', 0),
        ('conv_3x3_p_back', 2),
        ('conv_3x3_p_back', 3)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip3 = Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_3x3_p', 1),
        ('conv_3x3_n', 2), ('conv_3x3_n', 0),
        ('conv_3x3_p', 2), ('conv_5x5_p', 3),
        ('conv_5x5_n_back', 2),
        ('conv_3x3_p_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsc10_new_skip2 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 0), ('avg_pool_3x3_p', 1),
        ('avg_pool_3x3_p', 0), ('avg_pool_3x3_p', 1),
        ('conv_3x3_p', 2), ('avg_pool_3x3_n', 0),
        ('avg_pool_3x3_n_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip1 = Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_3x3_n', 1),
        ('conv_3x3_n', 2), ('conv_3x3_p', 1),
        ('conv_5x5_p', 1), ('conv_3x3_p', 2),
        ('conv_3x3_p_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip = Genotype(
    normal=[
        ('conv_3x3_n', 1), ('conv_3x3_p', 0),
        ('conv_3x3_p', 0), ('avg_pool_3x3_p', 1),
        ('conv_3x3_p', 2), ('conv_3x3_n', 0),
        ('conv_3x3_p_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new_base0 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 1), ('avg_pool_3x3_p', 0),
        ('avg_pool_3x3_n', 2), ('avg_pool_3x3_p', 1),
        ('avg_pool_3x3_n', 2), ('avg_pool_3x3_n', 3),
        ('avg_pool_3x3_n_back', 2),
        ('avg_pool_3x3_n_back', 3)],
    normal_concat=range(2, 5)
)

dvsc10_new_base1 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_5x5_n', 0),
        ('conv_5x5_p', 1), ('conv_3x3_p', 0),
        ('conv_5x5_n', 1), ('conv_3x3_p', 0),
        ('avg_pool_3x3_p_back', 2),
        ('conv_3x3_p_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsc10_new_base2 = Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_3x3_p', 1),
        ('conv_5x5_n', 1), ('avg_pool_3x3_p', 0),
        ('avg_pool_3x3_n', 3), ('conv_5x5_n', 1),
        ('avg_pool_3x3_n_back', 2),
        ('avg_pool_3x3_n_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new_base3 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 0), ('conv_5x5_p', 1),
        ('conv_3x3_p', 1), ('conv_3x3_n', 0),
        ('conv_5x5_p', 1), ('conv_3x3_n', 0),
        ('conv_3x3_p_back', 2),
        ('avg_pool_3x3_n_back', 3)],
    normal_concat=range(2, 5)
)

dvsc10_grad2 = Genotype(
    normal=[
        ('avg_pool_3x3_n', 1), ('conv_5x5_p', 0),
        ('conv_5x5_n', 1), ('conv_5x5_n', 0),
        ('conv_3x3_p', 3), ('conv_5x5_n', 1),
        ('conv_5x5_p_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_grad1 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 1), ('conv_5x5_p', 0),
        ('avg_pool_3x3_n', 2), ('avg_pool_3x3_n', 1),
        ('avg_pool_3x3_p', 2), ('conv_5x5_n', 1),
        ('conv_5x5_p_back', 2),
        ('conv_3x3_p_back', 3)],
    normal_concat=range(2, 5))

dvsg_new2 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 1), ('conv_5x5_p', 0),
        ('conv_3x3_p', 1), ('conv_3x3_p', 0),
        ('conv_3x3_p', 1), ('avg_pool_3x3_p', 0),
        ('avg_pool_3x3_n_back', 2),
        ('avg_pool_3x3_n_back', 3)],
    normal_concat=range(2, 5))

dvsg_new1 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 1), ('conv_5x5_p', 0),
        ('conv_3x3_p', 1), ('conv_3x3_p', 0),
        ('conv_3x3_p', 1),  ('avg_pool_3x3_p', 0),
        ('avg_pool_3x3_n_back', 2),
        ('conv_5x5_n_back', 3)],
    normal_concat=range(2, 5))

dvscal_new1 = Genotype(
    normal=[
        ('conv_5x5_n', 0), ('conv_5x5_n', 1),
        ('conv_5x5_n', 1), ('conv_5x5_p', 0),
        ('avg_pool_3x3_p', 1), ('conv_5x5_p', 0),
        ('avg_pool_3x3_n_back', 2),
        ('avg_pool_3x3_n_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new8 = Genotype(
    normal=[('conv_5x5_p', 0), ('conv_5x5_p', 1),
            ('conv_3x3_p', 0), ('conv_5x5_n', 1),
            ('conv_5x5_p', 0), ('conv_5x5_n', 1),
            ('avg_pool_3x3_n_back', 2),
            ('avg_pool_3x3_n_back', 3)],
    normal_concat=range(2, 5)
)

dvsc10_new7 = Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_5x5_p', 1),
        ('conv_3x3_p', 0), ('conv_5x5_n', 1),
        ('conv_5x5_p', 0), ('conv_5x5_n', 1),
        ('conv_3x3_n_back', 2),
        ('avg_pool_3x3_n_back', 2)],
    normal_concat=range(2, 5))

dvsc10_new6 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_3x3_p', 0),
        ('conv_3x3_p', 0), ('conv_3x3_p', 1),
        ('conv_3x3_p', 0), ('avg_pool_3x3_p', 1),
        ('avg_pool_3x3_n_back', 2),
        ('avg_pool_3x3_n_back', 2)],
    normal_concat=range(2, 5))

dvsc10_new5 = Genotype(
    normal=[
        ('conv_5x5_p', 1), ('conv_3x3_p', 0),
        ('conv_3x3_p', 0), ('conv_5x5_p', 1),
        ('conv_3x3_p', 0), ('avg_pool_3x3_p', 1),
        ('avg_pool_3x3_n_back', 2),
        ('avg_pool_3x3_n_back', 2)],
    normal_concat=range(2, 5))

dvsc10_new4 = Genotype(
    normal=[
        ('conv_3x3_n', 1), ('conv_3x3_p', 0),
        ('conv_5x5_p', 1), ('conv_5x5_p', 0),
        ('conv_5x5_p', 1), ('conv_5x5_p', 0),
        ('avg_pool_3x3_p_back', 2),
        ('avg_pool_3x3_n_back', 2)],
    normal_concat=range(2, 5),
)

dvsc10_new3 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 0), ('conv_3x3_n', 1),
        ('conv_3x3_n', 1), ('conv_3x3_n', 0),
        ('avg_pool_3x3_p', 2), ('conv_3x3_n', 1),
        ('avg_pool_3x3_n_back', 2),
        ('avg_pool_3x3_p', 2)],
    normal_concat=range(2, 5),
)

dvsc10_new2 = Genotype(normal=[
    ('conv_3x3_p', 0), ('conv_3x3_n', 1),
    ('conv_3x3_n', 1), ('avg_pool_3x3_p', 0),
    ('avg_pool_3x3_p', 2), ('conv_3x3_n', 1),
    ('avg_pool_3x3_n_back', 2),
    ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5),
)

dvsc10_new1 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('avg_pool_3x3_p', 0),
        ('avg_pool_3x3_p', 0), ('conv_3x3_n', 1),
        ('conv_3x3_p', 0), ('conv_3x3_p', 1),
        ('conv_3x3_p_back', 2),
        ('conv_3x3_n_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new0 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('avg_pool_3x3_p', 0),
        ('avg_pool_3x3_p', 2), ('conv_3x3_n', 1),
        ('conv_3x3_p', 0), ('conv_3x3_p', 3),
        #('conv_3x3_p_back', 2),
        #('conv_3x3_n_back', 3)
        ],
    normal_concat=range(2, 5)
)
cifar_new_skip1 = Genotype(
    normal=[
        ('conv_5x5_n', 0), ('conv_5x5_p', 1),
        ('avg_pool_3x3_p', 0), ('avg_pool_3x3_n', 2),
        ('avg_pool_3x3_p', 2), ('conv_5x5_p', 0),
        ('avg_pool_3x3_n_back', 2),
        ('avg_pool_3x3_p_back', 3)
    ],
    normal_concat=range(2, 5))

cifar_new1 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 1), ('avg_pool_3x3_p', 0),
        ('conv_3x3_n', 0), ('avg_pool_3x3_p', 1),
        ('avg_pool_3x3_p', 2), ('conv_3x3_p', 0),
        ('avg_pool_3x3_n_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5)
)

cifar_new2 = Genotype(
    normal=[
        ('conv_3x3_n', 0), ('avg_pool_3x3_p', 1),
        ('conv_3x3_p', 0), ('avg_pool_3x3_p', 1),
        ('conv_3x3_p', 2), ('conv_3x3_n', 0),
        ('conv_3x3_n_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5),
)

cifar_new0 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 1), ('avg_pool_3x3_n', 0),  # 2, 3
        ('conv_3x3_n', 0), ('avg_pool_3x3_p', 1),  # 4, 5
        ('conv_3x3_p', 2), ('conv_3x3_n', 3),  # 6 , 7
        ('avg_pool_3x3_n_back', 2),
        ('conv_3x3_p_back', 1)],
    normal_concat=range(2, 5)
)

# 新增：包含注意力机制的基因型
cifar_with_attention = Genotype(
    normal=[
        ('conv_3x3_p', 0), ('spike_self_attn_p', 1),  # 2, 3
        ('spike_cbam_p', 1), ('conv_5x5_n', 2),  # 4, 5
        ('spike_channel_attn_p', 2), ('conv_3x3_n', 3),  # 6, 7
        ('spike_self_attn_n_back', 2),
        ('spike_cbam_n_back', 3)],
    normal_concat=range(2, 5)
)

# 注意力增强的DVS基因型
dvsc10_with_attention = Genotype(
    normal=[
        ('spike_self_attn_p', 0), ('conv_3x3_p', 1),  # 2, 3
        ('spike_cbam_p', 1), ('conv_5x5_n', 2),  # 4, 5
        ('spike_channel_attn_p', 2), ('spike_spatial_attn_n', 3),  # 6, 7
        ('spike_self_attn_n_back', 2),
        ('spike_cbam_p_back', 3)],
    normal_concat=range(2, 5)
)

"""====== Quadratic Convolution Architectures ======"""

dvsc10_new_skip21 = Genotype(
    normal=[
        ('conv_3x3_n', 0), ('conv_5x5_p', 1),  # 2
        ('conv_3x3_p', 1), ('conv_5x5_p', 2),  # 3
        ('conv_5x5_n', 2), ('conv_3x3_p', 1),  # 4
        # ('conv_3x3_p_back', 2), ('conv_5x5_p_back', 2)
    ],
    normal_concat=range(2, 5)
)

# 纯二次卷积架构
dvsc10_quadratic_pure = Genotype(
    normal=[
        ('quad_conv_3x3_n', 0), ('quad_conv_5x5_p', 1),  # 2, 3
        ('quad_conv_3x3_p', 1), ('quad_conv_5x5_p', 2),  # 4, 5
        ('quad_conv_5x5_n', 2), ('quad_conv_3x3_p', 1),  # 6, 7
        #('quad_conv_3x3_n_back', 2), ('quad_conv_5x5_p_back', 3)
        ],
    normal_concat=range(2, 5)
)

# 混合二次卷积和普通卷积架构
dvsc10_quadratic_mixed = Genotype(
    normal=[
        ('quad_conv_3x3_p', 0), ('conv_3x3_p', 1),  # 2, 3
        ('conv_5x5_p', 1), ('quad_conv_3x3_p', 2),  # 4, 5
        ('quad_conv_5x5_p', 0), ('conv_3x3_p', 3),  # 6, 7
        ('conv_3x3_n_back', 2), ('quad_conv_3x3_p_back', 3)],
    normal_concat=range(2, 5)
)

# 负激活二次卷积架构
dvsc10_quadratic_negative = Genotype(
    normal=[
        ('quad_conv_3x3_n', 0), ('quad_conv_5x5_n', 1),  # 2, 3
        ('quad_conv_3x3_n', 1), ('quad_conv_5x5_n', 2),  # 4, 5
        ('quad_conv_5x5_n', 0), ('quad_conv_3x3_n', 3),  # 6, 7
        ('quad_conv_3x3_p_back', 2), ('quad_conv_5x5_n_back', 3)],
    normal_concat=range(2, 5)
)

# 标准二次卷积架构
dvsc10_quadratic_standard = Genotype(
    normal=[
        ('quad_conv_3x3', 0), ('quad_conv_5x5', 1),  # 2, 3
        ('quad_conv_3x3', 1), ('quad_conv_5x5', 2),  # 4, 5
        ('quad_conv_5x5', 0), ('quad_conv_3x3', 3),  # 6, 7
        #('quad_conv_3x3_back', 2), ('quad_conv_5x5_back', 3)
        ],
    normal_concat=range(2, 5)
)

# 二次卷积与注意力机制结合
dvsc10_quadratic_attention = Genotype(
    normal=[
        ('quad_conv_3x3_p', 0), ('spike_self_attn_p', 1),  # 2, 3
        ('spike_cbam_p', 1), ('quad_conv_5x5_p', 2),  # 4, 5
        ('quad_conv_3x3_p', 2), ('spike_spatial_attn_p', 3),  # 6, 7
        ('spike_self_attn_n_back', 2),
        ('quad_conv_5x5_p_back', 3)],
    normal_concat=range(2, 5)
)
